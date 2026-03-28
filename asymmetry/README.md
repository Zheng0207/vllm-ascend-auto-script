# vLLM Ascend 自动化基准测试工具

## 快速开始

### 1. 修改配置文件

编辑 `config.yaml`，配置你的测试参数：

```yaml
# 通用配置
common:
  model_path: "/path/to/your/model"      # 模型路径
  log_dir: "/path/to/logs"               # 日志目录
  master_addr: "80.48.33.145"            # 主节点IP
  master_port: "29500"                   # 主节点端口
  network_interface: "enp209s0f3"        # 网卡名称
  afd_port: 29666                        # AFD通信端口

# 基准测试参数
benchmark:
  bsize_list: "24 32 40 48 80 96"        # batch size 列表
  attn_arr: [12, 8, 4]                    # attention 卡数数组
  ffn_arr: [4, 4, 4]                      # ffn 卡数数组
  input_list: "4096 8192 16384"           # 输入长度列表
  ubatch_list: "2 3"                      # micro batch size 列表
  start_device: 0                         # 起始设备号
  data_multiplier: 16                     # 数据量倍数

# ais_bench 配置
ais_bench:
  config_py: "/path/to/synthetic_config.py"
  model_py: "/path/to/vllm_api_stream_chat.py"
  warmup_request_count: 1536
  formal_max_out_len: 1024
```

### 2. 运行测试

```bash
# 运行完整基准测试
./run_auto_benchmark.sh
```

### 3. 查看结果

测试结果保存在 `benchmark_results/` 目录下：

```
benchmark_results/
├── global_summary.csv                         # 全局汇总CSV
└── BSIZE_24_12A4F_UB2_IN4096_20240328_120000/ # 单次测试结果
    ├── result.yaml                            # 单次测试结果YAML
    ├── summary.txt                            # 文本格式报告
    ├── log/
    │   ├── warmup.log
    │   └── benchmark.log
    ├── profile/
    │   ├── ffn/
    │   └── attention/
    └── script/
        ├── run_params.txt
        └── config_snapshot_*.py
```

---

## 单次测试结果文件 (result.yaml)

每次测试完成后，会生成 `result.yaml` 文件，包含完整的测试配置和性能指标：

```yaml
# ==============================================================================
# 单次基准测试结果
# 生成时间: 2024-03-28 12:00:00
# ==============================================================================

test_config:
  batch_size: 24                    # 每卡batch size
  dp: 12                            # 数据并行度 (attention卡数)
  ubatch_size: 2                    # micro batch size
  attn_cnt: 12                      # attention卡数
  ffn_cnt: 4                        # ffn卡数
  global_batch_size: 288            # 全局batch size = dp * batch_size
  data_multiplier: 16               # 数据量倍数

request_config:
  input_len: 4096                   # 输入token长度
  output_len: 1024                  # 输出token长度
  warmup_request_count: 1536        # 预热请求数
  formal_request_count: 4608        # 正式测试请求数
  max_model_len: 7168               # = input_len + 3 * output_len

metrics:
  tpot_avg: 15.23                   # 平均TPOT (ms)
  tpot_min: 12.45                   # 最小TPOT (ms)
  tpot_max: 28.67                   # 最大TPOT (ms)
  tpot_med: 14.89                   # 中位数TPOT (ms)
  tpot_p75: 16.12                   # TPOT 75分位 (ms)
  tpot_p90: 18.34                   # TPOT 90分位 (ms)
  tpot_p99: 22.56                   # TPOT 99分位 (ms)
  out_token_throughput: 1234.56     # 输出token吞吐量 (token/s)
  throughput_per_die: 77.16         # 每卡吞吐量 (token/s)
```

---

## 自动计算参数

以下参数会根据配置自动计算，无需手动设置：

| 参数 | 计算公式 |
|------|----------|
| `global_batch_size` | `dp * batch_size` |
| `max_model_len` | `input_len + 3 * output_len` |
| `formal_request_count` | `data_multiplier * global_batch_size` |
| `throughput_per_die` | `out_token_throughput / (attn_cnt + ffn_cnt)` |

---

## 单独启动服务

如果只想单独启动 Attention 或 FFN 服务：

```bash
# 启动 Attention 服务
./attn.sh -d 4,5,6,7,8,9,10,11    # 指定设备号

# 启动 FFN 服务
./ffn.sh -d 0,1,2,3               # 指定设备号
```

### 命令行参数说明

| 参数 | 说明 | 默认值 (来自config.yaml) |
|------|------|--------------------------|
| `-d` | 设备号，逗号分隔 | attention: 4,5,6,7,8,9,10,11 / ffn: 0,1,2,3 |
| `-s` | batch size | 40 |
| `-n` | 设备数量 | 8 |
| `-c` | AFD size (如 8A4F) | 8A8F |
| `-u` | ubatch size | 2 |
| `-M` | max model len | 8192 (自动计算: input + 3*output) |
| `-l` | 日志目录 | /home/cyj/run_vllm_v13/log |
| `-m` | master addr | 80.48.33.145 |
| `-t` | master port | 29500 |
| `-i` | 网络接口 | enp209s0f3 |

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.yaml` | **主配置文件 - 所有参数在这里修改** |
| `yaml_parser.sh` | YAML解析器，自动被其他脚本调用 |
| `attn.sh` | Attention 服务启动脚本 |
| `ffn.sh` | FFN 服务启动脚本 |
| `run_auto_benchmark.sh` | 自动化基准测试主脚本 |
| `profileabc.py` | Profile 数据解析脚本 |

---

## 注意事项

1. **依赖要求**
   - Python 3 + PyYAML (`pip install pyyaml`)
   - Ascend Toolkit 环境
   - ais_bench 工具

2. **配置优先级**
   - 命令行参数 > config.yaml 配置

3. **测试流程**
   ```
   预热 -> 正式测试 -> 收集Profile -> 生成报告/YAML -> 清理进程
   ```

4. **多组测试**
   - `attn_arr` 和 `ffn_arr` 数组长度必须相同
   - 会遍历 `bsize_list x attn/ffn pairs x input_list x ubatch_list` 所有组合

---

## 示例：快速修改测试配置

```yaml
# 只测试一个 batch size
benchmark:
  bsize_list: "40"

# 测试不同的 attention/ffn 组合
benchmark:
  attn_arr: [8, 4]
  ffn_arr: [4, 2]

# 只测试短输入
benchmark:
  input_list: "4096 8192"
```

修改后直接运行 `./run_auto_benchmark.sh` 即可。
