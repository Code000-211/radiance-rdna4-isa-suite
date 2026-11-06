#!/usr/bin/env bash
# Builds real Radiance host components plus the CPU-only comparison candidates.
# Does not build/run an inference engine; does NOT emulate a GPU on the CPU.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ $# == 2 ]] || { echo "usage: $0 RADIANCE_SOURCE NEW_WORK_DIRECTORY" >&2; exit 2; }
SRC="$(realpath "$1")"; WORK="$(realpath -m "$2")"
[[ -f "$SRC/abi/rad_abi.h" && ! -e "$WORK" ]] || { echo 'source missing or output exists' >&2; exit 2; }
mkdir -p "$WORK/logs"
cmake -S "$HERE/replay/host-test" -B "$WORK/host" -DRADIANCE_SOURCE="$SRC" -DCMAKE_BUILD_TYPE=Release >"$WORK/logs/configure-host.txt" 2>&1
cmake --build "$WORK/host" -j "${RAD_PLANNER_BUILD_JOBS:-2}" >"$WORK/logs/build-host.txt" 2>&1
cmake -S "$HERE/planner/native" -B "$WORK/candidates" -DRADIANCE_SOURCE="$SRC" -DCMAKE_BUILD_TYPE=Release >"$WORK/logs/configure-candidates.txt" 2>&1
cmake --build "$WORK/candidates" -j "${RAD_PLANNER_BUILD_JOBS:-2}" >"$WORK/logs/build-candidates.txt" 2>&1
bash "$HERE/check-planner.sh" >"$WORK/logs/planner-tests.txt" 2>&1
bash "$HERE/planner.sh" cpu-demo --worker "$WORK/host/rad-kbench-replay" --libref "$WORK/host/libref.so" \
  --plugin "$WORK/candidates/planner_cpu.so" --out "$WORK/comparison" --execute-trusted-code >"$WORK/logs/comparison.txt" 2>&1
printf '%s\n' "CPU_PLANNER_EXPERIMENT_COMPLETE: $WORK/comparison/comparison.json (not GPU calibration)"
