#!/usr/bin/env bash
# Pure Python/static planning. Executes no native plugin and no GPU work.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ $# == 2 ]] || { echo "usage: $0 REVIEWED_RADIANCE_SOURCE NEW_OUTPUT_DIRECTORY" >&2; exit 2; }
SRC="$(realpath "$1")"; OUT="$(realpath -m "$2")"; EX="$HERE/planner/examples/v06"
[[ -d "$SRC/libr4d" && ! -e "$OUT" ]] || { echo 'source missing or output already exists' >&2; exit 2; }
mkdir -p "$OUT/logs"
bash "$HERE/check-planner.sh" >"$OUT/logs/planner-tests.txt" 2>&1
bash "$HERE/planner.sh" catalog-operators --source "$SRC" --out "$OUT/catalog.json"
bash "$HERE/planner.sh" plan-operators --catalog "$OUT/catalog.json" --workloads "$EX/operator-workloads.json" \
  --system "$EX/dual-r9700-unmeasured.json" --out "$OUT/operators"
bash "$HERE/planner.sh" plan-graph --catalog "$OUT/catalog.json" --workloads "$EX/operator-workloads.json" \
  --system "$EX/dual-r9700-unmeasured.json" --graph "$EX/tp2-moe-graph.json" --out "$OUT/graph"
bash "$HERE/planner.sh" sensitivity --catalog "$OUT/catalog.json" --workloads "$EX/operator-workloads.json" \
  --systems "$EX/rate-sensitivity.json" --out "$OUT/sensitivity.json"
bash "$HERE/planner.sh" model-workloads --source "$SRC" --config "$EX/qwen4exp-example-config.json" \
  --run "$EX/qwen4exp-example-run.json" --out "$OUT/partial-model-workloads.json"
bash "$HERE/planner.sh" plan-operators --catalog "$OUT/catalog.json" --workloads "$OUT/partial-model-workloads.json" \
  --system "$EX/dual-r9700-unmeasured.json" --out "$OUT/partial-model-plan"
# A deliberate strict-mode nonpass must remain nonpass, not an accidentally green gate.
set +e
bash "$HERE/planner.sh" plan-graph --catalog "$OUT/catalog.json" --workloads "$EX/operator-workloads.json" \
  --system "$EX/dual-r9700-unmeasured.json" --graph "$EX/tp2-moe-graph.json" --out "$OUT/strict-nonpass" \
  --require-calibrated >"$OUT/logs/strict-nonpass.txt" 2>&1
RC=$?
set -e
[[ "$RC" == 2 ]] || { echo "expected strict INCOMPLETE exit 2, got $RC" >&2; exit 1; }
printf '%s\n' "FLOW_SCENARIOS_AND_TESTS_COMPLETE: $OUT. No GPU/TP2 execution or calibration."
