#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python; [ -x "$PY" ] || PY=../.venv/bin/python
PYTHONPATH=src "$PY" -m armlab.cli run "$@"
