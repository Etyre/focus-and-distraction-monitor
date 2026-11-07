#!/bin/bash
# End-to-end self-test that never touches the running app or its data.
# Uses a throwaway data dir and log (FOCUS_MONITOR_DATA / FOCUS_MONITOR_LOG), does not start the
# web server (port 8790 belongs to the running app) and makes no model calls.
#   ./test.sh
# Steps: compile every module; unit-check the LLM preamble/contact wiring against shared/;
# run the collector for ~8 s against the temp dir (needs Screen Recording + Accessibility for
# the terminal, else that step just reports FAILED); classify once with --no-claude; print stats.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
# macOS has no coreutils `timeout`: run a command with a wall-clock limit.
run_to(){ local t=$1; shift; "$@" & local pid=$!; ( sleep "$t"; kill "$pid" 2>/dev/null ) & local wp=$!
  wait "$pid" 2>/dev/null; local rc=$?; kill "$wp" 2>/dev/null; wait "$wp" 2>/dev/null; return $rc; }
TMP=$(mktemp -d /tmp/focus-monitor-test.XXXXXX)
export FOCUS_MONITOR_DATA="$TMP/data" FOCUS_MONITOR_LOG="$TMP/test.log"
echo "temp data dir: $FOCUS_MONITOR_DATA"
fail=0

echo "--- compile"
$PY -m py_compile focus_monitor/*.py focus_monitor/classifiers/*.py build_app.py && echo ok || fail=1

echo "--- LLM standard: preamble + contact_eli + check (no network)"
$PY - <<'PYEOF' || fail=1
import sys
from focus_monitor import llm
pre = llm.llm_contact.preamble()
s = llm.with_preamble([{"type": "text", "text": "MY RULES", "cache_control": {"type": "ephemeral"}}])
assert s[0]["text"].startswith(pre) and s[0]["text"].endswith("MY RULES") and s[0]["cache_control"], s
assert llm.with_preamble(s)[0]["text"] == s[0]["text"], "preamble must not be added twice"
assert llm.with_preamble("x").startswith(pre)
assert llm.CONTACT_TOOL["name"] == "contact_eli"
hit = llm.llm_contact.extract({"content": [{"type": "text", "text": "fine.\n@ELI: hello"}]})
assert hit == [("marker", "hello")], hit
hit = llm.llm_contact.extract({"content": [{"type": "tool_use", "name": "contact_eli", "input": {"message": "hi"}}]})
assert hit == [("tool", "hi")], hit
assert llm.llm_contact.extract({"content": [{"type": "text", "text": '{"ok": true}'}]}) == []
print("ok: preamble prepended once, tool present, marker + tool detected, clean output passes")
PYEOF

echo "--- check (permissions; model lookup needs a key or network and may say FAILED)"
run_to 60 $PY -m focus_monitor check 2>&1 | grep -v "^$" | sed 's/^/  /'

echo "--- collector for 8 s into the temp dir"
run_to 8 $PY -m focus_monitor collect >"$TMP/collect.out" 2>&1
grep -c "" "$TMP/collect.out" >/dev/null && tail -3 "$TMP/collect.out" | sed 's/^/  /'
if [ -f "$FOCUS_MONITOR_DATA/focus.db" ]; then
  n=$($PY -c "import sqlite3,sys;c=sqlite3.connect(sys.argv[1]);print(c.execute('select count(*) from segments').fetchone()[0])" "$FOCUS_MONITOR_DATA/focus.db")
  echo "  segments recorded: $n"
else
  echo "  FAILED: no focus.db written (permissions?)"; fail=1
fi

echo "--- classify once, local models only"
run_to 60 $PY -m focus_monitor classify --once --no-claude 2>&1 | tail -2 | sed 's/^/  /' || fail=1

echo "--- stats"
$PY -m focus_monitor stats --days 1 2>&1 | sed 's/^/  /' || fail=1

echo "--- log written with timestamps?"
head -2 "$FOCUS_MONITOR_LOG" 2>/dev/null | sed 's/^/  /' || { echo "  no log"; fail=1; }

rm -rf "$TMP"
[ $fail = 0 ] && echo "PASS" || { echo "FAIL"; exit 1; }
