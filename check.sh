#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$HERE"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export RAD_ISA_REQUIRE_LLVM=1
export RAD_ISA_REQUIRE_CXX=1
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s replay_tests -v
