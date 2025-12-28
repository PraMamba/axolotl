#!/bin/bash
# Run all Ulysses + Ring-Attention validation tests (Phase 2.2)
#
# This script runs the complete test suite for multi-GPU validation.
# Requires 8 GPUs (runs smaller configs on subsets).
#
# Usage:
#   ./run_all_tests.sh
#
# Output:
#   outputs/baseline_no_cp/
#   outputs/ulysses_ring_hybrid_sp3_rp2/
#   outputs/ulysses_ring_ulysses_only_sp3/
#   outputs/ulysses_ring_ring_only_rp4/
#   outputs/ulysses_ring_manual_sp4_rp2/

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║          Ulysses + Ring-Attention Multi-GPU Validation (Phase 2.2)          ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check GPU availability
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${NUM_GPUS} GPUs"
echo ""

if [ "$NUM_GPUS" -lt 4 ]; then
    echo -e "${RED}ERROR: At least 4 GPUs required for validation tests${NC}"
    echo "Available GPUs: $NUM_GPUS"
    exit 1
fi

# Set NCCL timeout for distributed training
export NCCL_TIMEOUT=1800  # 30 minutes

# Function to run a single test
run_test() {
    local config=$1
    local num_processes=$2
    local test_name=$3

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${YELLOW}Running: $test_name ($num_processes GPUs)${NC}"
    echo "Config: $config"
    echo ""

    if [ "$NUM_GPUS" -lt "$num_processes" ]; then
        echo -e "${YELLOW}SKIP: Not enough GPUs (need $num_processes, have $NUM_GPUS)${NC}"
        echo ""
        return 0
    fi

    # Run training
    if accelerate launch --num-processes "$num_processes" \
        -m axolotl.cli.train "$config"; then
        echo -e "${GREEN}✅ PASS: $test_name${NC}"
    else
        echo -e "${RED}❌ FAIL: $test_name${NC}"
        return 1
    fi

    echo ""
}

# Track results
PASSED=0
FAILED=0
SKIPPED=0

# Test 1: Baseline (no context parallelism)
if run_test "examples/ulysses-ring-attn/config_baseline_no_cp.yml" 1 "Baseline (no CP)"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Test 2: Ulysses-only (3 GPUs)
if run_test "examples/ulysses-ring-attn/config_ulysses_only_sp4.yml" 3 "Ulysses-only (sp=3, rp=1)"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Test 3: Ring-only (4 GPUs)
if run_test "examples/ulysses-ring-attn/config_ring_only_rp4.yml" 4 "Ring-only (sp=1, rp=4)"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Test 4: Hybrid (6 GPUs)
if run_test "examples/ulysses-ring-attn/config_hybrid_sp3_rp2.yml" 6 "Hybrid (sp=3, rp=2)"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Test 5: Manual override (8 GPUs)
if run_test "examples/ulysses-ring-attn/config_manual_sp4_rp2.yml" 8 "Manual override (sp=4, rp=2)"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# Summary
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                              TEST SUMMARY                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo ""
    echo "Next step: Run validation script to compare results"
    echo "  python examples/ulysses-ring-attn/compare_runs.py"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Check logs in outputs/*/train.log${NC}"
    exit 1
fi
