#!/bin/bash

# ====================== 配置区 ======================
PT_SAVE_SCRIPT="./batch/test_sparse_flash_attention_pt_save.py"
PERF_SCRIPT="./perf_sparse_flash_attention.py"
ANALYZE_SCRIPT="./analyze_prof_kernel.py"
PT_DIR="./pt_files"
PROFILE_DIR="./prof_data"
ITERATIONS=100
KEYWORD="sparse_flash_attention"
# ====================================================

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --pt_file <路径>      指定单个 .pt 文件（跳过生成步骤）"
    echo "  --iterations <次数>   每个用例执行次数 (默认 $ITERATIONS)"
    echo "  --profile_dir <目录>  profile 输出目录 (默认 $PROFILE_DIR)"
    echo "  --keyword <关键字>    算子筛选关键字 (默认 $KEYWORD)"
    echo "  -h, --help            显示帮助"
    echo ""
    echo "示例:"
    echo "  $0                                    # 全流程：生成pt → 采集profile → 分析"
    echo "  $0 --pt_file ./pt_files/xxx.pt       # 跳过生成，只跑指定用例"
    echo "  $0 --iterations 200                  # 跑 200 次"
}

PT_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pt_file)       PT_FILE="$2";      shift 2 ;;
        --iterations)    ITERATIONS="$2";    shift 2 ;;
        --profile_dir)   PROFILE_DIR="$2";   shift 2 ;;
        --keyword)       KEYWORD="$2";       shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "未知参数: $1"; usage; exit 1 ;;
    esac
done

set -e

# ==================== 第一步：生成 pt 文件 ====================
if [ -n "$PT_FILE" ]; then
    echo "===== 跳过 pt 生成，使用指定文件: $PT_FILE ====="
    if [ ! -f "$PT_FILE" ]; then
        echo "错误: 文件不存在: $PT_FILE"
        exit 1
    fi
else
    echo "===== 第一步：生成 pt 文件 ====="
    python3 -m pytest -rA -s "$PT_SAVE_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    echo "pt 文件已生成到 $PT_DIR/"
fi

# ==================== 第二步：采集 profile ====================
echo ""
echo "===== 第二步：采集 profile ====="
rm -rf "$PROFILE_DIR"

run_perf() {
    local pt=$1
    echo ""
    echo "----- $pt -----"
    python3 "$PERF_SCRIPT" --pt_file "$pt" --iterations "$ITERATIONS" --profile_dir "$PROFILE_DIR"
}

if [ -n "$PT_FILE" ]; then
    run_perf "$PT_FILE"
else
    count=0
    for f in "$PT_DIR"/*.pt; do
        [ -f "$f" ] || continue
        count=$((count + 1))
        echo "[${count}] 处理: $f"
        run_perf "$f"
    done
    if [ "$count" -eq 0 ]; then
        echo "错误: $PT_DIR/ 下没有找到 .pt 文件"
        exit 1
    fi
fi

echo ""
echo "Profile 数据已保存到: $PROFILE_DIR"

# ==================== 第三步：分析 profile ====================
echo ""
echo "===== 第三步：分析 profile ====="
python3 "$ANALYZE_SCRIPT" --profile_dir "$PROFILE_DIR" --keyword "$KEYWORD"

echo ""
echo "===== 全部完成 ====="
