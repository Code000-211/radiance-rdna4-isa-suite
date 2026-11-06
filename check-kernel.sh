#!/usr/bin/env bash
# Lint one HIP source, assembly file or code object for gfx1201. See docs/CHECK.md.
#   bash check-kernel.sh FILE [--compile-commands build/compile_commands.json] [--kernel NAME] [--json OUT]
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m radisa check "$@"
