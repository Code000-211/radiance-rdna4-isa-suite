#!/usr/bin/env bash
# Lint many HIP sources / assembly files / code objects for gfx1201 in one run. See radisa/batch.py.
#   bash check-tree.sh --compile-commands build/compile_commands.json [--changed-since origin/main] \
#        [--jobs 8] [--timeout-per-file 1800] [--max-kernels-per-file 500] [--format sarif --out radisa.sarif]
#   bash check-tree.sh fixtures/check/clean_waits.s some/dir --include '\.(hip|s)$'
#   bash check-tree.sh --each FILE...    one check-kernel.sh run per file (the pre-commit hook);
#                                        set RADISA_COMPILE_COMMANDS to pass --compile-commands
# Exit: 0 clean, 1 a file FAILs (or --fail-on), 3 a file is INCOMPLETE, 2 bad invocation.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == "--each" ]]; then
  shift
  worst=0
  for f in "$@"; do
    extra=()
    if [[ -n "${RADISA_COMPILE_COMMANDS:-}" && "$f" == *.hip ]]; then
      extra=(--compile-commands "$RADISA_COMPILE_COMMANDS")
    fi
    rc=0
    bash "$HERE/check-kernel.sh" "$f" ${extra[@]+"${extra[@]}"} || rc=$?
    # 1 (FAIL) outranks 3 (INCOMPLETE) outranks 2 (bad invocation) outranks 0.
    if (( rc == 1 )) || { (( rc == 3 )) && (( worst != 1 )); } || { (( rc != 0 )) && (( worst == 0 )); }; then
      worst=$rc
    fi
  done
  exit "$worst"
fi
# Equivalent: `python3 -m radisa.batch` (the same parser and code, without the cli wiring).
exec python3 -m radisa check-all "$@"
