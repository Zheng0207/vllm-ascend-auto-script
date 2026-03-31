# DeepEP 自动化基准测试工具

## 快速开始

### 1. 修改配置文件

编辑 `config.yaml`，配置你的测试参数：

```yaml
# 通用配置
common:
  # 模型路径
  model_path: "/home/cyj/weight/DSV2LiteWeight"
  # 日志根目录
  log_dir: "/home/cyj/run_vllm_v13/log"
  # 本地IP地址
  local_ip: "80.48.33.145"
  # 网络接口名
  network_interface: "enp209s0f3"
  # HTTP服务端口
  port: 8006
  # 张量并行大小
  tp: 1
  # 专家并行配置 (支持多个值，会遍历测试)
  expert_per_rank:
    - 8
    - 4

# 基准测试参数
benchmark:
  # batch size 列表
  bsize_list: "48 80 96"
  # 数据并行大小列表 (支持多个值，会遍历测试)
  dp_list:
    - 8
    - 4
  # 输入长度 (StringConfig Input 的 MinValue/MaxValue)
  input_len: 4096
  # 数据量倍数
  data_multiplier: 16
  # 起始设备号
  start_device: 8

# ais_bench 配置
ais_bench:
  config_py: "/path/to/synthetic_config.py"
  model_py: "/path/to/vllm_api_stream_chat.py"
  # 预热配置
  warmup_request_count: 1536
  warmup_max_out_len: 2
  # 正式测试配置
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
├── global_summary.csv                          # 全局汇总CSV
└── BSIZE_48_DP8_IN4096_OUT1024_E8_20240331_120000/
    ├── script/                    # 配置快照
    ├── log/                       # 日志文件
    ├── profile/                   # Profile数据
    ├── summary.txt                # 测试报告
    └── result.yaml                # 结果YAML
```

---

## 配置说明

### 遍历参数

| 参数 | 说明 |
|------|------|
| `bsize_list` | 每卡 batch size 列表 |
| `dp_list` | 数据并行大小列表（设备数量）|
| `expert_per_rank` | 每rank专家数列表 |

### 自动计算参数

| 参数 | 计算公式 |
|------|----------|
| `total_batch` | `dp * bsize` |
| `request_count` | `data_multiplier * total_batch` |
| `devices` | `start_device` 到 `start_device + dp - 1` |

---

## StringConfig 配置 (ais_bench)

脚本会自动修改 `synthetic_config.py` 中的 StringConfig 参数：

```python
"StringConfig" : {
    "Input" : {
        "Method": "uniform",
        "Params": {"MinValue": 4096, "MaxValue": 4096}  # 固定输入长度
    },
    "Output" : {
        "Method": "gaussian",
        "Params": {"Mean": 1024, "MinValue": 1024, "MaxValue": 1024}  # 固定输出长度
    }
}
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.yaml` | **主配置文件 - 所有参数在这里修改** |
| `yaml_parser.sh` | YAML解析器，自动被其他脚本调用 |
| `decode_only.sh` | Decode Only 服务启动脚本 |
| `run_auto_benchmark.sh` | 自动化基准测试主脚本 |

---

## 输出指标

| 指标 | 说明 |
|------|------|
| `TPOT_AVG` | 平均TPOT (ms) |
| `TPOT_MIN/MAX/MED` | 最小/最大/中位数TPOT |
| `TPOT_P75/P90/P99` | TPOT分位数 |
| `OutTokenThru` | 输出token吞吐量 (token/s) |
| `Throughput_Per_Die` | 每卡吞吐量 (token/s) |
