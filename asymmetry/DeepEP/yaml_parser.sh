#!/bin/bash
# ==============================================================================
# YAML 配置解析器
# 用法: source yaml_parser.sh <yaml_file>
# ==============================================================================

if [ -z "$1" ]; then
    echo "用法: source yaml_parser.sh <yaml_file>"
    return 1 2>/dev/null || exit 1
fi

YAML_FILE="$1"

if [ ! -f "$YAML_FILE" ]; then
    echo "错误: YAML文件不存在: $YAML_FILE"
    return 1 2>/dev/null || exit 1
fi

# 使用Python解析YAML并生成bash变量
eval $(python3 -c "
import yaml
import sys

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            items.append((new_key, ' '.join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)

with open('$YAML_FILE', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

flat = flatten_dict(config)

for key, value in flat.items():
    bash_key = key.upper()
    if isinstance(value, str):
        safe_value = value.replace(\"'\", \"'\\''\")
        print(f\"{bash_key}='{safe_value}'\")
    else:
        print(f'{bash_key}=\"{value}\"')
")

# 设置兼容性别名
# common
export MODEL_PATH="$COMMON_MODEL_PATH"
export LOG_DIR="$COMMON_LOG_DIR"
export LOCAL_IP="$COMMON_LOCAL_IP"
export NETWORK_INTERFACE="$COMMON_NETWORK_INTERFACE"
export PORT="$COMMON_PORT"
export TP="$COMMON_TP"
export EXPERT_PER_RANK_LIST="$COMMON_EXPERT_PER_RANK"

# profiler
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_ENABLE="$PROFILER_MODEL_RUNNER_ENABLE"
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_WAIT="$PROFILER_MODEL_RUNNER_WAIT"
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_WARMUP="$PROFILER_MODEL_RUNNER_WARMUP"
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_ACTIVE="$PROFILER_MODEL_RUNNER_ACTIVE"
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_REPEAT="$PROFILER_MODEL_RUNNER_REPEAT"
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_SKIP_FIRST="$PROFILER_MODEL_RUNNER_SKIP_FIRST"
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_DIR="$PROFILER_MODEL_RUNNER_DIR"
export PROFILE_SRC="$PROFILER_MODEL_RUNNER_DIR"

# environment
export PYTHONPATH="$ENVIRONMENT_PYTHONPATH:$PYTHONPATH"
export HCCL_BUFFSIZE="$ENVIRONMENT_HCCL_BUFFSIZE"
export ASCEND_TOOLKIT_PATH="$ENVIRONMENT_ASCEND_TOOLKIT_PATH"

# benchmark
export BSIZE_LIST="$BENCHMARK_BSIZE_LIST"
export DP_LIST="$BENCHMARK_DP_LIST"
export INPUT_LEN_LIST="$BENCHMARK_INPUT_LEN_LIST"
export DATA_MULTIPLIER="$BENCHMARK_DATA_MULTIPLIER"
export START_DEVICE="$BENCHMARK_START_DEVICE"

# ais_bench
export CONFIG_PY="$AIS_BENCH_CONFIG_PY"
export MODEL_PY="$AIS_BENCH_MODEL_PY"
export WARMUP_REQUEST_COUNT="$AIS_BENCH_WARMUP_REQUEST_COUNT"
export WARMUP_MAX_OUT_LEN="$AIS_BENCH_WARMUP_MAX_OUT_LEN"
export FORMAL_MAX_OUT_LEN="$AIS_BENCH_FORMAL_MAX_OUT_LEN"
