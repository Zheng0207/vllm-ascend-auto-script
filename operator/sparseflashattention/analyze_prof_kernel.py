#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the software repository root for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""扫描 torch_npu profiler 输出目录，从 kernel_details.csv 中提取
npu_sparse_flash_attention 算子的耗时统计（min / max / p50 / p75 / p99）。"""

import argparse
import glob
import os
import statistics
import sys

import pandas as pd

KERNEL_KEYWORD = "sparse_flash_attention"
DURATION_COL = "Duration(us)"


def find_kernel_details_csv(profile_dir):
    # profiler 输出目录结构: {host}_{pid}_{ts}_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
    candidates = glob.glob(os.path.join(profile_dir, "**/kernel_details.csv"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(profile_dir, "**/kernel_detail.csv"), recursive=True)
    return candidates


def analyze(csv_path, keyword):
    print(f"读取: {csv_path}")
    df = pd.read_csv(csv_path)

    if DURATION_COL not in df.columns:
        print(f"错误: 找不到列 '{DURATION_COL}'，当前列: {list(df.columns)}")
        sys.exit(1)

    name_col = None
    for col in ("Name", "name", "Op Name", "OP Name"):
        if col in df.columns:
            name_col = col
            break
    if name_col is None:
        print(f"错误: 找不到算子名列，当前列: {list(df.columns)}")
        sys.exit(1)

    mask = df[name_col].astype(str).str.contains(keyword, case=False, na=False)
    matched = df.loc[mask, DURATION_COL].dropna().values.tolist()

    if not matched:
        print(f"未找到包含 '{keyword}' 的算子记录。")
        print(f"  算子名示例: {df[name_col].unique()[:10].tolist()}")
        sys.exit(1)

    matched_us = [float(v) for v in matched]
    matched_ms = [v / 1000.0 for v in matched_us]
    matched_sorted = sorted(matched_ms)

    n = len(matched_sorted)
    p50 = matched_sorted[int(n * 0.50)] if n > 0 else 0
    p75 = matched_sorted[int(n * 0.75)] if n > 0 else 0
    p99 = matched_sorted[min(int(n * 0.99), n - 1)] if n > 0 else 0

    print("=" * 60)
    print(f"算子筛选关键字: {keyword}")
    print(f"匹配记录数:     {n}")
    print(f"min:            {min(matched_ms):.4f} ms")
    print(f"max:            {max(matched_ms):.4f} ms")
    print(f"mean:           {statistics.mean(matched_ms):.4f} ms")
    print(f"p50 (median):   {p50:.4f} ms")
    print(f"p75:            {p75:.4f} ms")
    print(f"p99:            {p99:.4f} ms")
    if n > 1:
        print(f"stdev:          {statistics.stdev(matched_ms):.4f} ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="分析 profiler kernel_details.csv 中 SFA 算子耗时")
    parser.add_argument("--profile_dir", default="./prof_data", help="profiler 输出根目录")
    parser.add_argument("--keyword", default=KERNEL_KEYWORD, help="算子名筛选关键字")
    args = parser.parse_args()

    csv_files = find_kernel_details_csv(args.profile_dir)
    if not csv_files:
        print(f"未在 {args.profile_dir} 下找到 kernel_details.csv")
        sys.exit(1)

    for csv_path in csv_files:
        analyze(csv_path, args.keyword)


if __name__ == "__main__":
    main()
