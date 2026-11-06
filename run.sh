#!/usr/bin/env bash
set -euo pipefail
# Compatibility with v0.1: run.sh BUILD_DIR SOURCE_ROOT RESULT_DIR.
# Source is not scanned as proof of ISA legality; final binary evidence is used.
if [[ $# -ne 3 ]]; then echo "Usage: $0 BUILD_DIR SOURCE_ROOT NEW_RESULT_DIR" >&2; exit 2; fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
[[ -d "$2" ]] || { echo 'Source root does not exist' >&2; exit 2; }
[[ ! -e "$3" ]] || { echo 'Result directory already exists; choose a fresh directory' >&2; exit 2; }
mkdir -p -- "$3"
collect_status=0
python3 -m radisa collect "$1" --out "$3/evidence" || collect_status=$?
[[ -f "$3/evidence/manifest.json" ]] || exit 2
opts=()
[[ -z "${RAD_ISA_LLVM_LIBRARY:-}" ]] || opts+=(--llvm-library "$RAD_ISA_LLVM_LIBRARY")
[[ -z "${RAD_ISA_OBJDUMP:-}" ]] || opts+=(--objdump "$RAD_ISA_OBJDUMP")
status=0
python3 -m radisa audit "$3/evidence/manifest.json" --out "$3/analysis" "${opts[@]}" || status=$?
if [[ $collect_status -ne 0 && $status -eq 0 ]]; then exit 2; fi
exit "$status"
