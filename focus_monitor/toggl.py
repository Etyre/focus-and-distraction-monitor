"""Toggl Track sync: your time entries are the best available statement of intent."""
from __future__ import annotations

import base64
import datetime as dt
import json
import logging
import os
import time
import urllib.error
import urllib.request

log = logging.getLogger("toggl")
API = "https://api.track.toggl.com/api/v9"

SCHEMA = """
CREATE TABLE IF NOT EXISTS toggl_entries (
    id INTEGER PRIMARY KEY,
    start REAL NOT NULL,
    stop REAL,                       -- NULL while running
    description TEXT NOT NULL DEFAULT '',
    project TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    synced REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS toggl_start ON toggl_entries(start);
"""


def token() -> str | None:
    return os.environ.get("TOGGL_API_TOKEN") or None


def _get(path: str, tok: str):
    auth = base64.b64encode(f"{tok}:api_token".encode()).decode()
    req = urllib.request.Request(API + path, headers={"Authorization": "Basic " + auth})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _ts(s: str | None) -> float | None:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


class TogglSync:
    def __init__(self, conn):
        self.conn = conn
        conn.executescript(SCHEMA)
        self.projects: dict[int, str] = {}
        self.projects_at = 0.0
        self.last_sync = 0.0

    def _project_names(self, tok: str):
        if time.time() - self.projects_at > 3600:
            try:
                self.projects = {p["id"]: p["name"] for p in _get("/me/projects", tok)}
                self.projects_at = time.time()
            except Exception as e:
                log.warning("project fetch failed: %s", e)

    def sync(self, lookback_hours: float = 30, min_interval: float = 120) -> int:
        tok = token()
        if not tok or time.time() - self.last_sync < min_interval:
            return 0
        self.last_sync = time.time()
        self._project_names(tok)
        now = dt.datetime.now(dt.timezone.utc)
        q = (f"/me/time_entries?start_date={(now - dt.timedelta(hours=lookback_hours)).strftime('%Y-%m-%dT%H:%M:%SZ')}"
             f"&end_date={(now + dt.timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%SZ')}")
        try:
            entries = _get(q, tok)
        except urllib.error.HTTPError as e:
            log.error("toggl HTTP %s (bad token?)", e.code)
            return 0
        except Exception as e:
            log.warning("toggl sync failed: %s", e)
            return 0
        with self.conn:
            for e in entries:
                self.conn.execute(
                    """INSERT INTO toggl_entries(id, start, stop, description, project, tags, synced)
                       VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET start=excluded.start, stop=excluded.stop,
                       description=excluded.description, project=excluded.project, tags=excluded.tags, synced=excluded.synced""",
                    (e["id"], _ts(e["start"]), _ts(e.get("stop")), e.get("description") or "",
                     self.projects.get(e.get("project_id") or 0, ""), ", ".join(e.get("tags") or []), time.time()))
        return len(entries)


def entry_at(conn, ts: float) -> dict | None:
    r = conn.execute("SELECT * FROM toggl_entries WHERE start <= ? AND (stop IS NULL OR stop > ?) "
                     "ORDER BY start DESC LIMIT 1", (ts, ts)).fetchone()
    return dict(r) if r else None


def entry_started_near(conn, ts: float, window: float = 90) -> dict | None:
    """An entry that was started within `window` seconds of ts - a declared task change / focus start."""
    r = conn.execute("SELECT * FROM toggl_entries WHERE start BETWEEN ? AND ? ORDER BY start LIMIT 1",
                     (ts - window, ts + window)).fetchone()
    return dict(r) if r else None


def entries_between(conn, t0: float, t1: float) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM toggl_entries WHERE start < ? AND COALESCE(stop, ?) > ? ORDER BY start", (t1, time.time(), t0))]


def describe(e: dict | None) -> str:
    if not e:
        return "(no Toggl entry running)"
    bits = [f"“{e['description']}”" if e["description"] else "(untitled)"]
    if e["project"]:
        bits.append(f"project: {e['project']}")
    if e["tags"]:
        bits.append(f"tags: {e['tags']}")
    return " | ".join(bits)
