"""Performance benchmark for dispatch_gmm_combine_decode operator.

Compares FusionOp (fused) vs SmallOps (decomposed) across configurable
parameter matrices. Outputs profiling data and a summary CSV.

Usage:
    python bench_dispatch_gmm_combine.py --config bench_config.yaml
    python bench_dispatch_gmm_combine.py --config bench_config.yaml --rank 0
"""

import argparse
import csv
import gc
import os
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import yaml

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

torch.manual_seed(42)
torch_npu.npu.config.allow_internal_format = True


# ── Configuration ────────────────────────────────────────────


@dataclass
class BenchConfig:
    hidden_size: int
    intermediate_size: int
    expert_num: int
    top_k: int
    batch_sizes: List[int]
    ep_world_sizes: List[int]
    dtypes: List[str]
    iterations: int
    warmup: int
    with_graph: bool
    profile: bool
    profile_output_dir: str
    csv_output: str
    shared_expert_rank_num: int
    with_mc2_mask: bool
    enable_dynamic_bs: bool
    dynamic_eplb: bool


def load_config(path: str) -> BenchConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return BenchConfig(
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        expert_num=raw["expert_num"],
        top_k=raw["top_k"],
        batch_sizes=raw["batch_sizes"],
        ep_world_sizes=raw["ep_world_sizes"],
        dtypes=raw["dtypes"],
        iterations=raw["iterations"],
        warmup=raw["warmup"],
        with_graph=raw["with_graph"],
        profile=raw.get("profile", True),
        profile_output_dir=raw.get("profile_output_dir", "./bench_results/profile"),
        csv_output=raw.get("csv_output", "./bench_results/summary.csv"),
        shared_expert_rank_num=raw.get("shared_expert_rank_num", 0),
        with_mc2_mask=raw.get("with_mc2_mask", False),
        enable_dynamic_bs=raw.get("enable_dynamic_bs", False),
        dynamic_eplb=raw.get("dynamic_eplb", False),
    )


# ── Model definitions ────────────────────────────────────────


class DecodeMoeOps(torch.nn.Module):

    def __init__(self, gmm1_weight, gmm1_weight_scale, gmm2_weight,
                 gmm2_weight_scale, ep_hcomm_info, batch_size,
                 token_hidden_size, moe_intermediate_size, ep_world_size,
                 moe_expert_num, global_rank_id, shared_expert_rank_num=0,
                 dynamic_eplb=False):
        super().__init__()
        self.ep_hcomm_info = ep_hcomm_info
        self.batch_size = batch_size
        self.token_hidden_size = token_hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.ep_world_size = ep_world_size
        self.moe_expert_num = moe_expert_num
        self.global_rank_id = global_rank_id
        self.shared_expert_rank_num = shared_expert_rank_num
        is_shared_expert = global_rank_id < shared_expert_rank_num
        moe_expert_num_per_rank = (moe_expert_num //
                                   (ep_world_size - shared_expert_rank_num))
        self.local_expert_num = (1 if is_shared_expert
                                 else moe_expert_num_per_rank)
        self.ep_recv_count_size = self.local_expert_num * ep_world_size
        self.dynamic_eplb = dynamic_eplb
        self.gmm1_weight = torch.empty([
            self.local_expert_num, self.token_hidden_size,
            self.moe_intermediate_size * 2
        ])
        self.gmm1_weight_scale = torch.empty(
            [self.local_expert_num, self.moe_intermediate_size * 2])
        self.gmm2_weight = torch.empty([
            self.local_expert_num, self.moe_intermediate_size,
            self.token_hidden_size
        ])
        self.gmm2_weight_scale = torch.empty(
            [self.local_expert_num, self.token_hidden_size])
        self._process_weights_after_loading(gmm1_weight, gmm1_weight_scale,
                                            gmm2_weight, gmm2_weight_scale)

    def _process_weights_after_loading(self, gmm1_weight, gmm1_weight_scale,
                                       gmm2_weight, gmm2_weight_scale):
        gmm1_weight = torch_npu.npu_format_cast(
            gmm1_weight, torch_npu.Format.FRACTAL_NZ)
        gmm2_weight = torch_npu.npu_format_cast(
            gmm2_weight, torch_npu.Format.FRACTAL_NZ)
        self.gmm1_weight = torch.nn.Parameter(gmm1_weight,
                                              requires_grad=False)
        self.gmm1_weight_scale = torch.nn.Parameter(gmm1_weight_scale,
                                                    requires_grad=False)
        self.gmm2_weight = torch.nn.Parameter(gmm2_weight,
                                              requires_grad=False)
        self.gmm2_weight_scale = torch.nn.Parameter(gmm2_weight_scale,
                                                    requires_grad=False)
        self.gmm1_weight_scale_fp32 = torch.nn.Parameter(
            gmm1_weight_scale.float(), requires_grad=False)
        self.gmm2_weight_scale_fp32 = torch.nn.Parameter(
            gmm2_weight_scale.float(), requires_grad=False)

    def forward(self, x, expert_ids, smooth_scales, expert_scales,
                x_active_mask):
        return self._apply_ops(x, expert_ids, smooth_scales, expert_scales,
                               x_active_mask)


class SmallOps(DecodeMoeOps):

    def __init__(self, gmm1_weight, gmm1_weight_scale, gmm2_weight,
                 gmm2_weight_scale, ep_hcomm_info, batch_size,
                 token_hidden_size, moe_intermediate_size, ep_world_size,
                 moe_expert_num, global_rank_id, shared_expert_rank_num=0,
                 dynamic_eplb=False):
        super().__init__(gmm1_weight, gmm1_weight_scale, gmm2_weight,
                         gmm2_weight_scale, ep_hcomm_info, batch_size,
                         token_hidden_size, moe_intermediate_size,
                         ep_world_size, moe_expert_num, global_rank_id,
                         shared_expert_rank_num, dynamic_eplb)
        self.tp_hcomm_info = ""

    def _apply_ops(self, x, expert_ids, smooth_scales, expert_scales,
                   x_active_mask):
        outputs = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=expert_ids,
            expert_scales=expert_scales,
            x_active_mask=x_active_mask,
            group_ep=self.ep_hcomm_info,
            ep_world_size=self.ep_world_size,
            ep_rank_id=self.global_rank_id,
            moe_expert_num=self.moe_expert_num,
            group_tp=self.tp_hcomm_info,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=self.shared_expert_rank_num,
            quant_mode=2,
            global_bs=self.batch_size * self.ep_world_size,
            expert_token_nums_type=1,
        )
        (expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums,
         ep_send_counts, tp_send_counts, expand_scales) = outputs
        output_dtype = x.dtype

        y1_int32 = torch_npu.npu_grouped_matmul(
            x=[expand_x],
            weight=[self.gmm1_weight],
            split_item=3,
            group_list_type=1,
            group_type=0,
            group_list=expert_token_nums,
            output_dtype=torch.int32)[0]
        y1, y1_scale = torch_npu.npu_dequant_swiglu_quant(
            x=y1_int32,
            weight_scale=self.gmm1_weight_scale.to(torch.float32),
            activation_scale=dynamic_scales,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=expert_token_nums,
            activate_left=True,
            quant_mode=1,
        )
        y2 = torch_npu.npu_grouped_matmul(
            x=[y1],
            weight=[self.gmm2_weight],
            scale=[self.gmm2_weight_scale],
            per_token_scale=[y1_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_token_nums,
            output_dtype=output_dtype)[0]
        combine_output = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=y2,
            expert_ids=expert_ids,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_send_counts,
            expert_scales=expert_scales,
            x_active_mask=x_active_mask,
            group_ep=self.ep_hcomm_info,
            ep_world_size=self.ep_world_size,
            ep_rank_id=self.global_rank_id,
            moe_expert_num=self.moe_expert_num,
            tp_send_counts=tp_send_counts,
            expand_scales=expand_scales,
            group_tp=self.tp_hcomm_info,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=self.shared_expert_rank_num,
            global_bs=self.batch_size * self.ep_world_size)
        return (combine_output, expert_token_nums)


class FusionOp(DecodeMoeOps):

    def _apply_ops(self, x, expert_ids, smooth_scales, expert_scales,
                   x_active_mask):
        output = torch.ops._C_ascend.dispatch_gmm_combine_decode(
            x=x,
            expert_ids=expert_ids,
            gmm1_permuted_weight=self.gmm1_weight,
            gmm1_permuted_weight_scale=self.gmm1_weight_scale_fp32,
            gmm2_weight=self.gmm2_weight,
            gmm2_weight_scale=self.gmm2_weight_scale_fp32,
            expert_scales=expert_scales,
            expert_smooth_scales=smooth_scales,
            x_active_mask=x_active_mask,
            group_ep=self.ep_hcomm_info,
            ep_rank_size=self.ep_world_size,
            ep_rank_id=self.global_rank_id,
            moe_expert_num=self.moe_expert_num,
            shared_expert_num=1,
            shared_expert_rank_num=self.shared_expert_rank_num,
            quant_mode=0,
            global_bs=self.batch_size * self.ep_world_size)
        return output

    def _process_weights_after_loading(self, gmm1_weight, gmm1_weight_scale,
                                       gmm2_weight, gmm2_weight_scale):
        gmm1_weight = torch_npu.npu_format_cast(
            gmm1_weight, torch_npu.Format.FRACTAL_NZ)
        gmm2_weight = torch_npu.npu_format_cast(
            gmm2_weight, torch_npu.Format.FRACTAL_NZ)
        gmm1_weight_scale = gmm1_weight_scale.float()
        gmm2_weight_scale = gmm2_weight_scale.float()

        if self.dynamic_eplb:
            self.gmm1_weight = [
                w.clone() for w in gmm1_weight.unbind(dim=0)
            ]
            self.gmm1_weight_scale_fp32 = [
                w.clone() for w in gmm1_weight_scale.unbind(dim=0)
            ]
            self.gmm2_weight = [
                w.clone() for w in gmm2_weight.unbind(dim=0)
            ]
            self.gmm2_weight_scale_fp32 = [
                w.clone() for w in gmm2_weight_scale.unbind(dim=0)
            ]
        else:
            self.gmm1_weight = [gmm1_weight.clone()]
            self.gmm1_weight_scale_fp32 = [gmm1_weight_scale.clone()]
            self.gmm2_weight = [gmm2_weight.clone()]
            self.gmm2_weight_scale_fp32 = [gmm2_weight_scale.clone()]


# ── Data generation ──────────────────────────────────────────


def generate_inputs(batch_size, hidden_size, intermediate_size, ep_world_size,
                    expert_num, global_rank_id, top_k, dtype_str,
                    enable_dynamic_bs, with_mc2_mask, shared_expert_rank_num):
    is_shared_expert = global_rank_id < shared_expert_rank_num
    moe_expert_num_per_rank = (expert_num //
                               (ep_world_size - shared_expert_rank_num))
    actual_bs = (int(torch.randint(2 if with_mc2_mask else 1,
                                    batch_size, [1]).item())
                 if enable_dynamic_bs else batch_size)
    local_expert_num = 1 if is_shared_expert else moe_expert_num_per_rank

    gmm1_out_dim = intermediate_size * 2
    gmm2_out_dim = hidden_size

    # Input tensor
    x = torch.rand([actual_bs, hidden_size]) * 10 - 5

    # Expert IDs
    expert_ids = (torch.arange(global_rank_id * batch_size * top_k,
                               global_rank_id * batch_size * top_k +
                               actual_bs * top_k).to(torch.int32).view(
                                   actual_bs, top_k) % expert_num)

    # Weights
    if is_shared_expert:
        gmm1_weight = torch.ones([local_expert_num, hidden_size,
                                  gmm1_out_dim]).to(torch.int8) * 4
        gmm2_weight = torch.ones([local_expert_num, intermediate_size,
                                  gmm2_out_dim]).to(torch.int8) * 4
        gmm1_weight[:, :, ::2] *= -1
        gmm2_weight[:, :, ::2] *= -1
        gmm1_weight_scale = torch.ones([local_expert_num,
                                        gmm1_out_dim]) * 0.0015
        gmm2_weight_scale = torch.ones([local_expert_num,
                                        gmm2_out_dim]) * 0.0015
    else:
        gmm1_weight = torch.randint(-16, 16, [local_expert_num, hidden_size,
                                               gmm1_out_dim]).to(torch.int8)
        gmm2_weight = torch.randint(-16, 16, [local_expert_num,
                                               intermediate_size,
                                               gmm2_out_dim]).to(torch.int8)
        gmm1_weight_scale = (torch.rand([local_expert_num,
                                         gmm1_out_dim]) * 0.003 + 0.0015)
        gmm2_weight_scale = (torch.rand([local_expert_num,
                                         gmm2_out_dim]) * 0.003 + 0.0015)

    expert_scales = torch.rand(actual_bs, top_k)

    # Apply dtype
    use_bfloat16 = dtype_str == "bfloat16"
    if use_bfloat16:
        x = x.bfloat16()
        gmm1_weight_scale = gmm1_weight_scale.bfloat16()
        gmm2_weight_scale = gmm2_weight_scale.bfloat16()
    else:
        x = x.half()

    smooth_scales = None
    x_active_mask = None
    valid_token_num = actual_bs
    if with_mc2_mask:
        valid_token_num = int(torch.randint(1, actual_bs, [1]).item())
        x_active_mask = torch.cat(
            (torch.ones(valid_token_num),
             torch.zeros(actual_bs - valid_token_num))).bool()

    inputs = (x, expert_ids, smooth_scales, expert_scales, x_active_mask)
    weights = (gmm1_weight, gmm1_weight_scale, gmm2_weight, gmm2_weight_scale)
    return inputs, weights, actual_bs, valid_token_num


# ── Distributed setup ────────────────────────────────────────


def setup_distributed(local_rank_id, ep_world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="hccl",
                            rank=local_rank_id,
                            world_size=ep_world_size)
    ep_ranks = list(np.arange(0, ep_world_size))
    ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
    ep_group_small = dist.new_group(backend="hccl", ranks=ep_ranks)

    ep_hcomm_fused = (ep_group._get_backend(torch.device("npu"))
                      .get_hccl_comm_name(local_rank_id))
    ep_hcomm_small = (ep_group_small._get_backend(torch.device("npu"))
                      .get_hccl_comm_name(local_rank_id))
    return ep_hcomm_fused, ep_hcomm_small


def cleanup_distributed():
    dist.destroy_process_group()


# ── Latency measurement ──────────────────────────────────────


def measure_latency(model, inputs, warmup, iterations, device_id):
    """Run warmup + timed iterations, return list of per-iteration latencies (ms)."""
    # Warmup
    for _ in range(warmup):
        model(*inputs)
    torch_npu.npu.synchronize(device_id)

    latencies = []
    for _ in range(iterations):
        torch_npu.npu.synchronize(device_id)
        t0 = time.perf_counter()
        model(*inputs)
        torch_npu.npu.synchronize(device_id)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
    return latencies


def compute_stats(latencies):
    """Compute summary statistics from latency list (ms)."""
    arr = np.array(latencies)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


# ── Profiling helper ─────────────────────────────────────────


def build_profiler(output_dir):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1,
                                             repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            output_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    )


# ── Single benchmark run ─────────────────────────────────────


@torch.inference_mode()
def bench_one(local_rank_id, cfg: BenchConfig, batch_size, ep_world_size,
              dtype_str):
    """Benchmark one (bs, ep, dtype) combination on a single rank."""
    global_rank_id = local_rank_id
    device_id = local_rank_id % 16
    torch_npu.npu.set_device(device_id)

    ep_hcomm_fused, ep_hcomm_small = setup_distributed(
        local_rank_id, ep_world_size)
    torch_npu.npu.synchronize(device_id)

    params = (batch_size, cfg.hidden_size, cfg.intermediate_size,
              ep_world_size, cfg.expert_num, global_rank_id,
              cfg.shared_expert_rank_num)
    inputs_cpu, weights_cpu, actual_bs, valid_token_num = generate_inputs(
        *params, cfg.top_k, dtype_str, cfg.enable_dynamic_bs,
        cfg.with_mc2_mask, cfg.shared_expert_rank_num)

    inputs = [d.npu() if d is not None else None for d in inputs_cpu]
    weights = [d.npu() if d is not None else None for d in weights_cpu]

    results = []

    for op_name, OpClass, ep_hcomm in [
        ("SmallOps", SmallOps, ep_hcomm_small),
        ("FusionOp", FusionOp, ep_hcomm_fused),
    ]:
        torch.manual_seed(42)
        model = OpClass(*weights, ep_hcomm, *params,
                        cfg.dynamic_eplb).npu()

        if cfg.with_graph:
            import torchair
            config = torchair.CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            model = torch.compile(model, backend=npu_backend)

        # Measure latency
        latencies = measure_latency(model, inputs, cfg.warmup, cfg.iterations,
                                    device_id)
        stats = compute_stats(latencies)
        throughput = (actual_bs * cfg.top_k / (stats["mean"] / 1000)
                      if stats["mean"] > 0 else 0)

        result = {
            "op_type": op_name,
            "batch_size": actual_bs,
            "ep_world_size": ep_world_size,
            "dtype": dtype_str,
            "expert_num": cfg.expert_num,
            "top_k": cfg.top_k,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "graph_mode": cfg.with_graph,
            "iterations": cfg.iterations,
            **stats,
            "throughput_tokens_per_s": throughput,
        }
        results.append(result)

        # Profiling pass
        if cfg.profile and local_rank_id == 0:
            prof_dir = os.path.join(
                cfg.profile_output_dir,
                f"{op_name}/bs{actual_bs}_ep{ep_world_size}_{dtype_str}")
            prof = build_profiler(prof_dir)
            with prof:
                model(*inputs)
                prof.step()

        del model
        gc.collect()
        torch.npu.empty_cache()

    cleanup_distributed()
    torch.npu.reset_peak_memory_stats()
    return results


def run_worker(local_rank_id, cfg_dict, batch_size, ep_world_size, dtype_str,
               result_queue):
    """Entry point for mp.spawn workers."""
    cfg = BenchConfig(**cfg_dict)
    try:
        results = bench_one(local_rank_id, cfg, batch_size, ep_world_size,
                            dtype_str)
        if local_rank_id == 0:
            result_queue.put(results)
    except Exception as e:
        if local_rank_id == 0:
            result_queue.put([{"error": str(e)}])


# ── CSV output ───────────────────────────────────────────────

CSV_COLUMNS = [
    "op_type", "batch_size", "ep_world_size", "dtype", "expert_num", "top_k",
    "hidden_size", "intermediate_size", "graph_mode", "iterations",
    "mean", "min", "max", "p50", "p95", "p99", "throughput_tokens_per_s",
]


def write_csv(results: List[dict], csv_path: str):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS,
                                extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in results:
            writer.writerow(row)


# ── CLI & main loop ──────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark dispatch_gmm_combine_decode operator")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file")
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=None,
                        help="Override batch_sizes from config")
    parser.add_argument("--ep-world-sizes", nargs="*", type=int, default=None,
                        help="Override ep_world_sizes from config")
    parser.add_argument("--dtypes", nargs="*", default=None,
                        help="Override dtypes from config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Allow CLI overrides
    batch_sizes = args.batch_sizes or cfg.batch_sizes
    ep_world_sizes = args.ep_world_sizes or cfg.ep_world_sizes
    dtypes = args.dtypes or cfg.dtypes

    # Cartesian product of all parameter combos
    combos = list(product(batch_sizes, ep_world_sizes, dtypes))
    total = len(combos)
    print(f"{'='*60}")
    print(f" dispatch_gmm_combine_decode Benchmark")
    print(f" {total} parameter combinations to test")
    print(f" {cfg.iterations} iterations + {cfg.warmup} warmup each")
    print(f"{'='*60}\n")

    # Clear previous CSV
    if Path(cfg.csv_output).exists():
        Path(cfg.csv_output).unlink()

    result_queue = mp.get_context("spawn").SimpleQueue()

    for idx, (bs, ep, dtype) in enumerate(combos, 1):
        tag = f"[{idx}/{total}] bs={bs}, ep={ep}, dtype={dtype}"
        print(f"{tag} ... ", end="", flush=True)

        cfg_dict = cfg.__dict__.copy()
        try:
            mp.spawn(
                run_worker,
                args=(cfg_dict, bs, ep, dtype, result_queue),
                nprocs=ep,
                join=True,
            )
            results = result_queue.get()
            if "error" in results[0]:
                print(f"ERROR: {results[0]['error']}")
                continue

            write_csv(results, cfg.csv_output)
            fused = [r for r in results if r["op_type"] == "FusionOp"][0]
            small = [r for r in results if r["op_type"] == "SmallOps"][0]
            speedup = (small["mean"] / fused["mean"]
                       if fused["mean"] > 0 else float("inf"))
            print(f"FusionOp {fused['mean']:.2f}ms | "
                  f"SmallOps {small['mean']:.2f}ms | "
                  f"speedup {speedup:.2f}x")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nResults saved to {cfg.csv_output}")
    if cfg.profile:
        print(f"Profiling data saved to {cfg.profile_output_dir}")


if __name__ == "__main__":
    main()
