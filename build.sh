#!/bin/bash
# Build dist/Focus Monitor.app (the signed launcher bundle wrapping this project's .venv).
# Idempotent: rebuilds in place. Does not touch a running copy; relaunch with
#   .venv/bin/python -m focus_monitor restart
# or ./install.sh. Set FOCUS_MONITOR_DIST=/some/dir to build somewhere else (e.g. a test build).
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "no .venv - see README"; exit 1; }
.venv/bin/python build_app.py
