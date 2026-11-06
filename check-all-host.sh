#!/usr/bin/env bash
# Full reproducible HOST test pipeline. Does not start the inference engine or a GPU.
# Requirements: Python 3.10+, LLVM disassembler, CMake, GNU/Clang C++20, clang++ ASan.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ $# == 2 ]] || { echo "usage: $0 RADIANCE_SOURCE NEW_WORK_DIRECTORY" >&2; exit 2; }
SRC="$(realpath "$1")"; WORK="$(realpath -m "$2")"
[[ -f "$SRC/abi/rad_abi.h" && ! -e "$WORK" ]] || { echo 'source missing or work directory exists' >&2; exit 2; }
command -v clang++ >/dev/null || [[ -x /opt/rocm/llvm/bin/clang++ ]] || { echo 'clang++ required for instrumented host checks' >&2; exit 2; }
# Prefer the toolchain Radiance itself builds with (ROCm's clang, docs/GUIDE.md); a stock
# clang older than 16 miscompiles the ASan probe, so refuse it instead of failing in a build.
CLANGXX="${RAD_REPLAY_ASAN_CXX:-}"
for candidate in "${ROCM_PATH:-/opt/rocm}/llvm/bin/clang++" /usr/lib/llvm-19/bin/clang++ \
                 /usr/lib/llvm-18/bin/clang++ /usr/lib/llvm-17/bin/clang++ /usr/lib/llvm-16/bin/clang++; do
    [[ -n "$CLANGXX" ]] && break
    [[ -x "$candidate" ]] && CLANGXX="$candidate"
 done
[[ -n "$CLANGXX" ]] || CLANGXX="$(command -v clang++ || true)"
[[ -n "$CLANGXX" ]] || { echo 'clang++ required for instrumented host checks' >&2; exit 2; }
clang_major="$("$CLANGXX" -dumpversion 2>/dev/null | cut -d. -f1)"
[[ "$clang_major" =~ ^[0-9]+$ ]] || clang_major=0
[[ "$clang_major" -ge 16 ]] || { echo "ASan checks need clang 16 or newer; $CLANGXX is $( "$CLANGXX" -dumpversion 2>/dev/null )" >&2; exit 2; }
mkdir -p -- "$WORK/logs"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
cmake -S "$HERE/replay/host-test" -B "$WORK/normal" -DRADIANCE_SOURCE="$SRC" -DCMAKE_BUILD_TYPE=Release > "$WORK/logs/configure-host.txt" 2>&1
cmake --build "$WORK/normal" -j "${RAD_REPLAY_BUILD_JOBS:-2}" > "$WORK/logs/build-host.txt" 2>&1
bash "$HERE/check.sh" > "$WORK/logs/python-tests.txt" 2>&1
export RAD_REPLAY_TEST_BUILD="$WORK/normal" RAD_REPLAY_TEST_LIBREF="$WORK/normal/libref.so"
python3 -m unittest discover -s "$HERE/replay_tests/native" -v > "$WORK/logs/native-tests.txt" 2>&1
cmake -S "$HERE/replay/host-test" -B "$WORK/asan" -DRADIANCE_SOURCE="$SRC" -DCMAKE_CXX_COMPILER="$CLANGXX" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS_RELWITHDEBINFO='-O1 -g -DNDEBUG' -DRAD_REPLAY_HOST_ASAN=ON > "$WORK/logs/configure-asan.txt" 2>&1
cmake --build "$WORK/asan" --target rad-kbench-replay replay-test-plugin -j "${RAD_REPLAY_BUILD_JOBS:-2}" > "$WORK/logs/build-asan.txt" 2>&1
export RAD_REPLAY_ASAN_BUILD="$WORK/asan"
python3 -m unittest discover -s "$HERE/replay_tests/asan" -v > "$WORK/logs/asan-tests.txt" 2>&1
python3 "$HERE/examples/boundary_corpus.py" --out "$WORK/corpus" --plugin "$WORK/normal/libref.so"
python3 -m radreplay run-corpus --expect "$WORK/corpus/expected.json" --worker "$WORK/normal/rad-kbench-replay" \
  --plugin "p0=$WORK/normal/libref.so" --out "$WORK/corpus-runs" --execute-trusted-code --warmups 1 --reps 3 > "$WORK/logs/corpus.txt" 2>&1
mapfile -d '' REPORTS < <(find "$WORK/corpus-runs" -mindepth 2 -maxdepth 2 -name report.json -print0 | sort -z)
opts=();for report in "${REPORTS[@]}"; do opts+=(--report "$report");done
python3 -m radreplay qualify --expect "$WORK/corpus/expected.json" "${opts[@]}" --profile host-contracts \
  --numeric-policy "$WORK/corpus/numeric-policy.json" --out "$WORK/qualification"
"$WORK/normal/rad-replay-inventory" "$WORK/normal/libref.so" "$WORK/rows.json"
python3 - "$HERE" "$WORK" <<'PY'
import json,sys
from pathlib import Path
from radreplay.qualification import row_coverage
w=Path(sys.argv[2]);cases=[p.parent/'manifest.json' for p in (w/'corpus').glob('*/manifest.json')]
reports=sorted((w/'corpus-runs').glob('*/report.json'))
code,r=row_coverage(w/'rows.json',cases,reports,w/'row-coverage')
# An intentionally GEMM-only corpus cannot claim all libref rows: INCOMPLETE is correct.
assert code==2 and r['tested_rows']==1 and r['declared_rows']>1,r
(w/'HOST_VALIDATION.json').write_text(json.dumps({'schema':'radiance-host-validation/1','status':'HOST_PIPELINE_PASS',
    'fixture_cases':len(cases),'tested_plugin_rows':r['tested_rows'],'declared_plugin_rows':r['declared_rows'],
    'gpu_execution':False,'tp2_execution':False,'whole_engine_built':False},indent=2)+'\n')
PY
printf '%s\n' "HOST_PIPELINE_PASS: $WORK/HOST_VALIDATION.json (not GPU/TP2 certification)"
