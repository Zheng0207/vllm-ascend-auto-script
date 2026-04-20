# SparseFlashAttention 性能测试

## 前置条件

- Ascend NPU 环境（Atlas A2/A3 系列），已安装 CANN、torch_npu、torchair
- 自定义算子已编译安装（`torch_npu.npu_sparse_flash_attention` 可用）

## 一键执行

```bash
cd <项目路径>/attention/sparse_flash_attention/tests/pytest

# 全流程：生成 pt → 采集 profile → 分析
bash perf_run.sh

# 跳过 pt 生成，只跑指定用例
bash perf_run.sh --pt_file ./pt_files/xxx.pt

# 自定义参数
bash perf_run.sh --iterations 200 --profile_dir ./my_prof
```

### `perf_run.sh` 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pt_file` | 指定 .pt 文件，跳过生成步骤 | 不指定则生成全部 |
| `--iterations` | 每个用例执行次数 | 100 |
| `--profile_dir` | profile 输出目录 | `./prof_data` |
| `--keyword` | 分析时算子名筛选关键字 | `sparse_flash_attention` |

## 分步执行

### 第一步：生成测试数据（.pt 文件）

```bash
python3 -m pytest -rA -s ./batch/test_sparse_flash_attention_pt_save.py -v -m ci
```

生成完成后，`./pt_files/` 目录下会产出若干 `.pt` 文件。

### 第二步：采集 profile

```bash
python3 perf_sparse_flash_attention.py --pt_file ./pt_files/<用例文件>.pt
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pt_file` | 测试数据 .pt 文件路径（必填） | - |
| `--iterations` | 执行次数 | 100 |
| `--profile_path` | profile 输出目录，传空字符串禁用 | `./prof_data` |

```bash
# 自定义迭代次数
python3 perf_sparse_flash_attention.py --pt_file ./pt_files/xxx.pt --iterations 200

# 不采集 profile
python3 perf_sparse_flash_attention.py --pt_file ./pt_files/xxx.pt --profile_path ""
```

### 第三步：分析 profile

```bash
python3 analyze_prof_kernel.py --profile_dir ./prof_data
```

从 `kernel_details.csv` 中筛选 SFA 算子记录，输出耗时统计：

```
============================================================
算子筛选关键字: sparse_flash_attention
匹配记录数:     100
min:            0.1234 ms
max:            0.1567 ms
mean:           0.1302 ms
p50 (median):   0.1289 ms
p75:            0.1345 ms
p99:            0.1523 ms
stdev:          0.0056 ms
============================================================
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--profile_dir` | profiler 输出根目录 | `./prof_data` |
| `--keyword` | 算子名筛选关键字 | `sparse_flash_attention` |

## 脚本执行流程

1. 从 Excel 读取参数，生成 .pt 测试数据文件
2. torchair graph 模式编译算子（`torch.compile` + aclgraph static kernel）
3. Warmup 5 次（排除编译和内存分配开销）
4. 采集 NPU profile，执行 N 次调用
5. 解析 `kernel_details.csv`，输出 min / max / mean / p50 / p75 / p99 耗时统计

## 查看详细 profile

```bash
python3 -m torch_npu.npu.profiler_analysis ./prof_data
```
