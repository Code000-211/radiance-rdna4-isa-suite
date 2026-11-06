#!/usr/bin/env bash
# Tests real Radiance registry/device/comparison code plus host-only negative fixtures.
# Usage: check-replay-host.sh /path/to/radiance /path/to/libref.so /fresh/build/dir
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# != 3 ]]; then echo "usage: $0 RADIANCE_SOURCE LIBREF_SO BUILD_DIR" >&2; exit 2; fi
SRC="$(realpath "$1")"; LIB="$(realpath "$2")"; BUILD="$(realpath -m "$3")"
[[ -f "$LIB" && -f "$SRC/abi/rad_abi.h" ]] || { echo "required source/plugin missing" >&2; exit 2; }
cmake -S "$ROOT/replay/host-test" -B "$BUILD" -DRADIANCE_SOURCE="$SRC" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j "${RAD_REPLAY_BUILD_JOBS:-2}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RAD_REPLAY_TEST_BUILD="$BUILD" RAD_REPLAY_TEST_LIBREF="$LIB"
python3 -m unittest discover -s "$ROOT/replay_tests/native" -v
