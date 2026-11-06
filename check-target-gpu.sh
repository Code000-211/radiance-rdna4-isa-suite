#!/usr/bin/env bash
# One already-approved single-plugin gfx1201 corpus, strict static gate then
# diagnostic dispatch tracing. No engine startup, no reset and no retries.
# The worker MUST be an AQL build with BOTH optional AQL patches.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ $# == 5 ]] || { echo "usage: $0 EXPECTED_CORPUS AQL_WORKER PLUGIN_ID PLUGIN_SO NEW_OUTPUT" >&2; exit 2; }
EXPECT="$(realpath "$1")"; WORKER="$(realpath "$2")"; ID="$3"; PLUGIN="$(realpath "$4")"; OUT="$(realpath -m "$5")"
[[ ! -e "$OUT" ]] || { echo 'fresh output required' >&2; exit 2; }
mkdir -p -- "$OUT"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
# Include the worker itself so runtime support kernels are part of the whitelist.
python3 -m radisa collect "$PLUGIN" "$WORKER" --out "$OUT/isa-inputs"
python3 -m radisa audit "$OUT/isa-inputs/manifest.json" --out "$OUT/isa-analysis"
python3 -m radreplay run-corpus --expect "$EXPECT" --worker "$WORKER" --plugin "$ID=$PLUGIN" \
  --out "$OUT/diagnostics" --execute-trusted-code --diagnostic-only --trace-dispatches \
  --isa-manifest "$OUT/isa-inputs/manifest.json" --isa-report "$OUT/isa-analysis/report.json"
printf '%s\n' 'Declared diagnostic corpus completed. GPU read instrumentation, TP2, raw tape and production calibration remain separate/unproved.'
