#!/usr/bin/env bash
# Harness drift-check script.
# Run:  bash script/check.sh
# Each gate prints PASS or FAIL; non-zero exit on any failure.

set -uo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
pass=0
fail=0

check() {
    local desc="$1"; shift
    if "$@" > /dev/null 2>&1; then
        echo -e "  ${GREEN}PASS${NC}  $desc"
        ((pass++))
    else
        echo -e "  ${RED}FAIL${NC}  $desc"
        ((fail++))
    fi
}

echo "=== Gate A: File existence (CLAUDE.md ↔ repo drift) ==="
# Manually kept list of source files referenced in CLAUDE.md's architecture tree.
# If you add/remove a source file, update both CLAUDE.md and this list.
expected_files=(
    "src/CMakeLists.txt"
    "src/sampleConfig.h.in"
    "src/ods/CMakeLists.txt"
    "src/ods/outlier_detection.h"
    "src/ods/outlier_detection.cpp"
    "src/ods/outlier_detection.cu"
    "src/ods/pipeline.h"
    "src/ods/pipeline.cpp"
    "src/ods/bvh.h"
    "src/ods/bvh.cpp"
    "src/ods/grid.h"
    "src/ods/grid.cpp"
    "src/ods/verify.h"
    "src/ods/verify.cpp"
    "src/ods/aabb.cu"
    "src/ods/state.h"
    "src/ods/timer.h"
    "script/run.py"
    "script/draw.py"
)
for f in "${expected_files[@]}"; do
    check "source file $f" test -f "$f"
done

echo ""
echo "=== Gate B: Compile-time invariant guards ==="
check "DIMENSION guard in outlier_detection.h" \
    grep -q '#error.*DIMENSION must be' src/ods/outlier_detection.h
check "OPTIMIZATION guard in outlier_detection.h" \
    grep -q '#error.*OPTIMIZATION must be' src/ods/outlier_detection.h
check "COMPACTION guard in outlier_detection.h" \
    grep -q '#error.*COMPACTION' src/ods/outlier_detection.h
check "MK guard in outlier_detection.h" \
    grep -q '#error.*MK must be' src/ods/outlier_detection.h
check "CMake DIMENSION check" \
    grep -q 'DIMENSION must be' src/ods/CMakeLists.txt
check "CMake OPTIMIZATION check" \
    grep -q 'OPTIMIZATION must be' src/ods/CMakeLists.txt

echo ""
echo "=== Gate C: Runtime invariant guards ==="
check "validate_params called from main" \
    grep -q 'validate_params' src/ods/outlier_detection.cpp
check "window % slide guard" \
    grep -q 'window.*divisible by.*slide' src/ods/outlier_detection.cpp
check "K <= MK guard" \
    grep -q 'K.*must be.*MK' src/ods/outlier_detection.cpp

echo ""
echo "=== Gate D: ADR coverage ==="
expected_adrs=(
    "001-custom-primitives-instead-of-triangles.md"
    "002-ray-bvh-inversion.md"
    "003-compile-time-flags-over-runtime-config.md"
    "004-fixqueue-static-array.md"
    "005-rebuild-over-update.md"
)
for adr in "${expected_adrs[@]}"; do
    check "ADR $adr" test -f "docs/decisions/$adr"
done

echo ""
echo "=== Gate E: No forbidden patterns ==="
# Using namespace std in headers is bad practice
if grep -rl "using namespace std" src/ods/*.h > /dev/null 2>&1; then
    echo -e "  ${RED}FAIL${NC}  No 'using namespace std' in headers"
    ((fail++))
else
    echo -e "  ${GREEN}PASS${NC}  No 'using namespace std' in headers"
    ((pass++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Results: ${GREEN}${pass} passed${NC}, ${RED}${fail} failed${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$fail" -gt 0 ]; then
    exit 1
fi