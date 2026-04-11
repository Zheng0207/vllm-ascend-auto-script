# dispatch_gmm_combine_decode 性能测试

对 `dispatch_gmm_combine_decode` 融合算子与拆分小算子（SmallOps）进行性能对比测试，支持多组 BS / EP / dtype 参数自动遍历，输出 profiling 数据和汇总 CSV。

## 环境要求

- 昇腾 NPU 环境（A3 卡）
- Python >= 3.8
- 依赖包：`torch`, `torch_npu`, `torchair`, `numpy`, `pyyaml`

安装依赖（如未安装）：

```bash
pip install pyyaml
# torch / torch_npu / torchair 请按昇腾环境文档安装
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `bench_config.yaml` | 测试参数配置文件 |
| `bench_dispatch_gmm_combine.py` | 主测试脚本 |
| `run_bench.sh` | 启动脚本 |

## 快速开始

### 1. 默认配置运行

```bash
bash run_bench.sh
```

### 2. 指定配置文件

```bash
bash run_bench.sh -c my_config.yaml
```

### 3. 命令行覆盖参数

```bash
# 只测 bs=64 和 bs=128，ep=4 和 ep=8，dtype=bfloat16
bash run_bench.sh --bs 64 128 --ep 4 8 --dt bfloat16

# 组合使用
bash run_bench.sh -c bench_config.yaml --bs 32 64 --ep 2 4
```

### 4. 直接用 Python 运行

```bash
python bench_dispatch_gmm_combine.py --config bench_config.yaml

# 带 CLI 覆盖
python bench_dispatch_gmm_combine.py \
    --config bench_config.yaml \
    --batch-sizes 64 128 \
    --ep-world-sizes 4 8 \
    --dtypes bfloat16
```

## 配置文件说明

编辑 `bench_config.yaml` 自定义测试参数：

```yaml
# 模型维度
hidden_size: 7168          # 隐藏层大小
intermediate_size: 2048    # 中间层大小
expert_num: 64             # 专家总数
top_k: 8                   # Top-K 路由数

# 测试矩阵（脚本自动遍历笛卡尔积）
batch_sizes: [16, 32, 64, 128, 256]
ep_world_sizes: [1, 2, 4, 8, 16]
dtypes: [bfloat16, int8]   # bfloat16 / int8

# 运行参数
iterations: 100            # 每组正式迭代次数
warmup: 10                 # 预热次数（不计入统计）
with_graph: true           # 是否入图（torch.compile）

# Profiling
profile: true              # 是否采集 profiling
profile_output_dir: "./bench_results/profile"

# 输出
csv_output: "./bench_results/summary.csv"
```

## 输出

### 1. 终端输出

运行时实时打印每组参数的对比结果：

```
============================================================
 dispatch_gmm_combine_decode Benchmark
 50 parameter combinations to test
 100 iterations + 10 warmup each
============================================================

[1/50] bs=16, ep=1, dtype=bfloat16 ... FusionOp 2.34ms | SmallOps 5.67ms | speedup 2.42x
[2/50] bs=16, ep=1, dtype=int8 ... FusionOp 1.98ms | SmallOps 4.89ms | speedup 2.47x
...
```

### 2. CSV 文件

路径：`bench_results/summary.csv`

| 列名 | 说明 |
|------|------|
| `op_type` | SmallOps / FusionOp |
| `batch_size` | 实际 batch size |
| `ep_world_size` | EP 并行度 |
| `dtype` | 数据类型 |
| `mean` | 平均延迟 (ms) |
| `min` | 最小延迟 (ms) |
| `max` | 最大延迟 (ms) |
| `p50` / `p95` / `p99` | 百分位延迟 (ms) |
| `throughput_tokens_per_s` | 吞吐量 (tokens/s) |

### 3. Profiling 数据

路径：`bench_results/profile/<op_type>/bs<bs>_ep<ep>_<dtype>/`

可用 TensorBoard 或 msprof 工具分析。

## 注意事项

- 脚本通过 `mp.spawn` 启动多进程，每个 EP rank 对应一张 NPU 卡，请确保可用卡数 >= `ep_world_size` 的最大值
- 默认配置 `ep_world_sizes: [1, 2, 4, 8, 16]`，需要至少 16 张卡；如果卡数不足，请在配置文件或命令行中调小 `ep_world_sizes`
- 首次入图编译耗时较长，属于正常现象
- `bench_results/` 目录可随时删除，不影响下次运行
