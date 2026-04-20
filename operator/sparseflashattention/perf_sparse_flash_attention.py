#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the software repository root for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import os
import sys

import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig

sys.path.insert(0, os.path.dirname(__file__))
from batch.sparse_flash_attention_process import Network


WARMUP_ITERS = 5
DEFAULT_ITERS = 100
DEVICE_ID = 0


def load_test_data(pt_file):
    return torch.load(pt_file, map_location="cpu")


def load_inputs_to_npu(input_dict):
    def to_npu(value):
        if isinstance(value, torch.Tensor):
            return value.npu()
        return value
    return {key: to_npu(value) for key, value in input_dict.items()}


def compile_model():
    torch._dynamo.reset()
    npu_model = Network().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    config.experimental_config.topology_sorting_strategy = "StableRDFS"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    npu_model = torch.compile(npu_model, fullgraph=True, backend=npu_backend, dynamic=False)
    return npu_model


def build_call_kwargs(npu_inputs, input_dict):
    return dict(
        query=npu_inputs["query"],
        key=npu_inputs["key"],
        value=npu_inputs["value"],
        sparse_indices=npu_inputs["sparse_indices"],
        scale_value=input_dict["scale_value"],
        block_table=npu_inputs.get("block_table"),
        actual_seq_lengths_query=npu_inputs.get("actual_seq_lengths_query"),
        actual_seq_lengths_kv=npu_inputs.get("actual_seq_lengths_kv"),
        query_rope=npu_inputs.get("query_rope"),
        key_rope=npu_inputs.get("key_rope"),
        sparse_block_size=input_dict.get("sparse_block_size", 1),
        layout_query=input_dict.get("layout_query", "BSND"),
        layout_kv=input_dict.get("layout_kv", "BSND"),
        sparse_mode=input_dict.get("sparse_mode", 3),
        pre_tokens=input_dict.get("pre_tokens", (1 << 63) - 1),
        next_tokens=input_dict.get("next_tokens", (1 << 63) - 1),
        attention_mode=input_dict.get("attention_mode", 2),
        return_softmax_lse=input_dict.get("return_softmax_lse", False),
    )


def run_perf(pt_file, iterations, profile_path):
    print(f"加载测试数据: {pt_file}")
    test_data = load_test_data(pt_file)
    params = test_data["params"]
    input_dict = test_data["input"]

    print(f"用例参数: {params[:12]}")
    torch_npu.npu.set_device(DEVICE_ID)

    npu_inputs = load_inputs_to_npu(input_dict)
    kwargs = build_call_kwargs(npu_inputs, input_dict)

    print("编译图模型 (torchair) ...")
    npu_model = compile_model()

    # warmup
    print(f"Warmup {WARMUP_ITERS} 次 ...")
    for _ in range(WARMUP_ITERS):
        npu_model(**kwargs)
        torch.npu.synchronize()

    # 采集 profile
    if profile_path:
        print(f"采集 profile + 执行 {iterations} 次 ...")
        import torch_npu.profiler as profiler
        os.makedirs(profile_path, exist_ok=True)
        experimental_config = profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1,
            aic_metrics=profiler.AiCMetrics.PipeUtilization,
            data_simplification=False,
        )
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.NPU,
            ],
            schedule=profiler.schedule(wait=0, warmup=0, active=iterations, repeat=1),
            on_trace_ready=profiler.tensorboard_trace_handler(profile_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            experimental_config=experimental_config,
        ) as prof:
            for _ in range(iterations):
                npu_model(**kwargs)
                prof.step()
        torch.npu.synchronize()
        print(f"Profile 数据已保存到: {profile_path}")
    else:
        print(f"执行 {iterations} 次 ...")
        for _ in range(iterations):
            npu_model(**kwargs)
        torch.npu.synchronize()
        print("执行完成（未采集 profile）")


def main():
    parser = argparse.ArgumentParser(description="SparseFlashAttention 性能测试 (graph mode)")
    parser.add_argument("--pt_file", required=True, help="测试数据 .pt 文件路径")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERS, help="执行次数 (默认 100)")
    parser.add_argument("--profile_path", default="./prof_data", help="profile 输出目录 (默认 ./prof_data，传空字符串禁用)")
    args = parser.parse_args()

    if not os.path.exists(args.pt_file):
        print(f"错误: 文件不存在: {args.pt_file}")
        sys.exit(1)

    profile_path = args.profile_path if args.profile_path else None
    run_perf(args.pt_file, args.iterations, profile_path)


if __name__ == "__main__":
    main()
