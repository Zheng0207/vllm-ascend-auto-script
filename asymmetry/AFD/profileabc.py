#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description="分析包含特定字符串的NPU Profiler目录")
parser.add_argument("work_dir", help="工作目录路径")
parser.add_argument("pattern", help="目录名必须包含的字符串（例如 202602240624）")
args = parser.parse_args()

# 设置工作目录
work_dir = args.work_dir
pattern = args.pattern

# 切换到工作目录
os.chdir(work_dir)

# 查找所有包含"20260119025"的目录
target_dirs = []
for item in os.listdir("."):
    if os.path.isdir(item) and pattern in item:
        target_dirs.append(item)

print(f"找到 {len(target_dirs)} 个匹配的目录:")
for d in target_dirs:
    print(f"  - {d}")

# 导入analyse函数并处理每个目录
from torch_npu.profiler.profiler import analyse

for dir_path in target_dirs:
    print(f"\n正在分析: {dir_path}")
    try:
        analyse(dir_path)
        print(f"完成分析: {dir_path}")
    except Exception as e:
        print(f"分析 {dir_path} 时出错: {e}")
