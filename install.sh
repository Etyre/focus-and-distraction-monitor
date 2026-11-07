#!/bin/bash
# Install (or reinstall) Focus Monitor. Idempotent; safe to rerun.
#
# Default: builds dist/Focus Monitor.app and registers it as a Login Item (the menu bar eye),
# then relaunches it if it is running so the new code is picked up.
#   ./install.sh
# Headless alternative under launchd (no menu bar icon) - mutually exclusive with the Login Item:
#   ./install.sh --launchd
#
# Config: ./config.toml   Secrets: Keychain service FocusMonitor   Log: ~/Library/Logs/focus-and-distraction-monitor.log
set -euo pipefail
cd "$(dirname "$0")"
LABEL=com.elityre.focus-and-distraction-monitor
PLIST_SRC="launchd/$LABEL.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
APP="$PWD/dist/Focus Monitor.app"

[ -x .venv/bin/python ] || { echo "no .venv - run: /opt/homebrew/bin/python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
[ -f config.toml ] || { cp config.example.toml config.toml; echo "created config.toml from config.example.toml - edit [profile] about_me etc."; }
mkdir -p "$HOME/Library/Logs" "$HOME/Library/Application Support/FocusMonitor"

if [ "${1:-}" = "--launchd" ]; then
  # Make sure the Login Item variant is not also running.
  osascript -e 'tell application "System Events" to delete login item "Focus Monitor"' 2>/dev/null || true
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  cp "$PLIST_SRC" "$PLIST_DST"
  launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
  echo "installed $LABEL (headless). Log: ~/Library/Logs/focus-and-distraction-monitor.log"
  exit 0
fi

# Default: the signed menu bar app as a Login Item.
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true   # never run both variants
./build.sh
osascript -e 'tell application "System Events" to delete login item "Focus Monitor"' 2>/dev/null || true
osascript -e "tell application \"System Events\" to make login item at end with properties {path:\"$APP\", hidden:false}" >/dev/null
if pgrep -f "Focus Monitor.app/Contents/MacOS/FocusMonitor" >/dev/null; then
  .venv/bin/python -m focus_monitor restart || true     # the running app relaunches itself with the new code
else
  open "$APP"
fi
echo "installed: Login Item 'Focus Monitor' -> $APP"
