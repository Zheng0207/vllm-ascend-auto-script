#!/bin/bash
# Launch script for dispatch_gmm_combine_decode benchmark.
#
# Usage:
#   bash run_bench.sh                              # default config
#   bash run_bench.sh -c my_config.yaml            # custom config
#   bash run_bench.sh -c config.yaml --bs 64 128   # override batch sizes
#   bash run_bench.sh -c config.yaml --ep 4 8      # override ep sizes
#   bash run_bench.sh -c config.yaml --dt bfloat16  # override dtype

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/bench_config.yaml"

# Defaults
BATCH_SIZES=""
EP_SIZES=""
DTYPES=""

usage() {
    echo "Usage: $0 [-c CONFIG] [--bs BS...] [--ep EP...] [--dt DT...]"
    echo ""
    echo "Options:"
    echo "  -c CONFIG   Path to YAML config file (default: bench_config.yaml)"
    echo "  --bs BS     Override batch_sizes (space-separated)"
    echo "  --ep EP     Override ep_world_sizes (space-separated)"
    echo "  --dt DT     Override dtypes (space-separated)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c) CONFIG="$2"; shift 2 ;;
        --bs)
            shift
            BS_LIST=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                BS_LIST+=("$1"); shift
            done
            BATCH_SIZES="${BS_LIST[*]}"
            ;;
        --ep)
            shift
            EP_LIST=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                EP_LIST+=("$1"); shift
            done
            EP_SIZES="${EP_LIST[*]}"
            ;;
        --dt)
            shift
            DT_LIST=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                DT_LIST+=("$1"); shift
            done
            DTYPES="${DT_LIST[*]}"
            ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

CMD="python ${SCRIPT_DIR}/bench_dispatch_gmm_combine.py --config ${CONFIG}"
[[ -n "$BATCH_SIZES" ]] && CMD+=" --batch-sizes ${BATCH_SIZES}"
[[ -n "$EP_SIZES" ]] && CMD+=" --ep-world-sizes ${EP_SIZES}"
[[ -n "$DTYPES" ]] && CMD+=" --dtypes ${DTYPES}"

echo "Running: ${CMD}"
echo ""
exec ${CMD}
