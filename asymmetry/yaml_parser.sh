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
            # 数组转换为bash数组格式
            items.append((new_key, ' '.join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)

with open('$YAML_FILE', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

flat = flatten_dict(config)

# 输出bash变量
for key, value in flat.items():
    # 转换为大写并替换特殊字符
    bash_key = key.upper()
    if isinstance(value, str):
        # 转义单引号
        safe_value = value.replace(\"'\", \"'\\''\")
        print(f\"{bash_key}='{safe_value}'\")
    else:
        print(f'{bash_key}=\"{value}\"')
")

# 设置一些兼容性别名 (保持与原脚本的变量名兼容)
# common
export MODEL_PATH="$COMMON_MODEL_PATH"
export LOG_DIR="$COMMON_LOG_DIR"
export MASTER_ADDR="$COMMON_MASTER_ADDR"
export MASTER_PORT="$COMMON_MASTER_PORT"
export NETWORK_INTERFACE="$COMMON_NETWORK_INTERFACE"
export AFD_PORT="$COMMON_AFD_PORT"
export EXPERT_PER_RANK_LIST="$COMMON_EXPERT_PER_RANK"

# attention
export ATTN_PORT="$ATTENTION_PORT"

# profiler
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_ENABLE="$PROFILER_MODEL_RUNNER_ENABLE"
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_WAIT="$PROFILER_MODEL_RUNNER_WAIT"
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_WARMUP="$PROFILER_MODEL_RUNNER_WARMUP"
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_ACTIVE="$PROFILER_MODEL_RUNNER_ACTIVE"
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_REPEAT="$PROFILER_MODEL_RUNNER_REPEAT"
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_SKIP_FIRST="$PROFILER_MODEL_RUNNER_SKIP_FIRST"
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_DIR="$PROFILER_MODEL_RUNNER_DIR"

export VLLM_ASCEND_FFN_PROFILER_ENABLE="$PROFILER_FFN_ENABLE"
export VLLM_ASCEND_FFN_PROFILER_WAIT="$PROFILER_FFN_WAIT"
export VLLM_ASCEND_FFN_PROFILER_WARMUP="$PROFILER_FFN_WARMUP"
export VLLM_ASCEND_FFN_PROFILER_ACTIVE="$PROFILER_FFN_ACTIVE"
export VLLM_ASCEND_FFN_PROFILER_REPEAT="$PROFILER_FFN_REPEAT"
export VLLM_ASCEND_FFN_PROFILER_SKIP_FIRST="$PROFILER_FFN_SKIP_FIRST"
export VLLM_ASCEND_FFN_PROFILER_DIR="$PROFILER_FFN_DIR"

# profile (使用 profiler 的 dir 作为源目录)
export PROFILE_FFN_SRC="$PROFILER_FFN_DIR"
export PROFILE_ATTN_SRC="$PROFILER_MODEL_RUNNER_DIR"
export PROFILE_PARSER_SCRIPT="$PROFILE_PARSER_SCRIPT"

# environment
export PYTHONPATH="$ENVIRONMENT_PYTHONPATH:$PYTHONPATH"
export HCCL_BUFFSIZE="$ENVIRONMENT_HCCL_BUFFSIZE"
export ASCEND_TOOLKIT_PATH="$ENVIRONMENT_ASCEND_TOOLKIT_PATH"
