#!/usr/bin/env bash
# Build and run the hazard probes on a gfx1201 GPU (Radeon AI PRO R9700). UNTESTED: written where no
# ROCm and no GPU exist; the build and launch steps have never been executed.
#
#   bash hwprobe/run_probes.sh [OUT_DIR]          (default hwprobe/build)
#
# Environment: ROCM_PATH (default /opt/rocm), HWPROBE_CLANG (device compiler, default
# $ROCM_PATH/llvm/bin/clang++), HIPCC (host compiler, default $ROCM_PATH/bin/hipcc),
# HWPROBE_ITERATIONS (launches per kernel, default 200), HWPROBE_GRID (workgroups, default 512),
# HWPROBE_CASE (only cases whose name contains it), HWPROBE_ANY_GPU=1 (allow a non-gfx1201 device).
#
# Writes OUT_DIR/results.json (host_runner) and OUT_DIR/lint/<case>.json (radisa check of the very
# code objects that ran). Exit: 0 every fixed variant matched (hazard variants are informational),
# 1 a fixed variant mismatched, 2 build, setup or runtime error.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
OUT=${1:-$HERE/build}
ROCM=${ROCM_PATH:-/opt/rocm}
CLANG=${HWPROBE_CLANG:-$ROCM/llvm/bin/clang++}
HIPCC=${HIPCC:-$ROCM/bin/hipcc}
ITER=${HWPROBE_ITERATIONS:-200}
GRID=${HWPROBE_GRID:-512}
fail() { echo "run_probes: $*" >&2; exit 2; }

[ -x "$CLANG" ] || fail "device compiler $CLANG not found (set HWPROBE_CLANG or ROCM_PATH)"
[ -x "$HIPCC" ] || fail "hipcc $HIPCC not found (set HIPCC or ROCM_PATH)"
python3 "$HERE/generate.py" --check || fail "hwprobe/cases is stale: run python3 hwprobe/generate.py"
mkdir -p "$OUT/hsaco" "$OUT/lint" || fail "cannot create $OUT"

# Device code: one raw gfx1201 code object (ELF, not an offload bundle) per case, for hipModuleLoad.
for src in "$HERE"/cases/*.hip; do
  name=$(basename "$src" .hip)
  "$CLANG" -x hip --offload-arch=gfx1201 --offload-device-only --no-gpu-bundle-output -nogpuinc -nogpulib \
    -O3 "$src" -o "$OUT/hsaco/$name.hsaco" || fail "compiling $src"
  # The linter's verdict on the same bytes that run (exit 1 on the hazard variants is expected).
  PYTHONPATH="$REPO" python3 -m radisa check "$OUT/hsaco/$name.hsaco" --json "$OUT/lint/$name.json" >/dev/null
  [ $? -le 1 ] || fail "radisa check could not analyse $OUT/hsaco/$name.hsaco"
done

python3 "$HERE/generate.py" --data "$OUT/data" --hsaco-dir "$OUT/hsaco" --grid "$GRID" >/dev/null \
  || fail "writing probe data"
"$HIPCC" -O2 -std=c++17 "$HERE/host_runner.cpp" -o "$OUT/host_runner" || fail "building host_runner"

"$OUT/host_runner" "$OUT/data/cases.tsv" "$ITER" "$OUT/results.json" ${HWPROBE_CASE:+"$HWPROBE_CASE"}
rc=$?
echo "results: $OUT/results.json  lint: $OUT/lint/"
case $rc in
  0) echo "fixed variants: all matched. Hazard variants without mismatches do NOT show a rule is unnecessary." ;;
  1) echo "a FIXED variant mismatched: harness bug (or, for d16_other_half, evidence against the linter's relaxation)" ;;
  *) rc=2 ;;
esac
exit $rc
