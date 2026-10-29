"""Toggl Track sync: your time entries are the best available statement of intent."""
from __future__ import annotations

import base64
import datetime as dt
import json
import logging
import os
import re
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
        self.last_deep = 0.0
        self.quota_until = 0.0   # Toggl's free plan has an HOURLY API quota (HTTP 402); wait it out

    def _project_names(self, tok: str):
        if time.time() - self.projects_at > 6 * 3600:
            try:
                self.projects = {p["id"]: p["name"] for p in _get("/me/projects", tok)}
                self.projects_at = time.time()
            except urllib.error.HTTPError as e:
                if e.code == 402:
                    self._quota_hit(e)
                else:
                    log.warning("project fetch failed: %s", e)
            except Exception as e:
                log.warning("project fetch failed: %s", e)

    def _quota_hit(self, e: urllib.error.HTTPError) -> None:
        body = e.read().decode(errors="replace") if hasattr(e, "read") else ""
        m = re.search(r"reset in (\d+) seconds", body)
        wait = int(m.group(1)) if m else 900
        self.quota_until = time.time() + wait + 30
        log.warning("toggl: hourly API quota hit - pausing sync for %d s", wait)

    DEEP_HOURS = 7 * 24       # entries corrected days later are picked up by the periodic deep sync
    DEEP_EVERY = 6 * 3600

    def sync(self, lookback_hours: float = 30, min_interval: float = 300) -> int:
        """Periodic sync: the last `lookback_hours` every call (edits land within minutes), and
        every DEEP_EVERY a DEEP_HOURS window, so entries you fix days later are corrected too.
        Every 5 min, not 2: the free plan's hourly quota is small, and manual re-syncs need room."""
        if not token() or time.time() - self.last_sync < min_interval:
            return 0
        deep = time.time() - self.last_deep > self.DEEP_EVERY
        r = self.resync(self.DEEP_HOURS if deep else lookback_hours)
        if r is not None and deep:
            self.last_deep = time.time()
        return r["fetched"] if r else 0

    def resync(self, lookback_hours: float) -> dict | None:
        """Fetch every entry in the window and make the local table match Toggl exactly: edited
        entries are updated, entries deleted in Toggl are removed. Everything downstream (segment
        intent, span classification, the dashboard) reads this table live, so corrections show up
        at once - but judgments the evaluator already made with the old entries are not re-run."""
        tok = token()
        if not tok:
            return None
        if time.time() < self.quota_until:
            log.info("toggl: skipping sync, API quota resets in %.0f s", self.quota_until - time.time())
            return None
        self.last_sync = time.time()
        self._project_names(tok)
        now = dt.datetime.now(dt.timezone.utc)
        t0 = now - dt.timedelta(hours=lookback_hours)
        q = (f"/me/time_entries?start_date={t0.strftime('%Y-%m-%dT%H:%M:%SZ')}"
             f"&end_date={(now + dt.timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%SZ')}")
        try:
            entries = _get(q, tok)
        except urllib.error.HTTPError as e:
            if e.code == 402:
                self._quota_hit(e)
            else:
                log.error("toggl HTTP %s (bad token?)", e.code)
            return None
        except Exception as e:
            log.warning("toggl sync failed: %s", e)
            return None
        changed = 0
        touched: list[tuple[float, float]] = []   # time ranges whose intent statement changed
        with self.conn:
            before = {r["id"]: (r["start"], r["stop"], r["description"], r["project"], r["tags"]) for r in
                      self.conn.execute("SELECT * FROM toggl_entries WHERE start >= ?", (t0.timestamp(),))}
            # For an entry that was still running when we last looked, judgments made since then
            # assumed it ran until at least that sync - so that is the 'old end' for touch purposes.
            synced_at = {r["id"]: r["synced"] for r in
                         self.conn.execute("SELECT id, synced FROM toggl_entries WHERE stop IS NULL")}
            seen = set()
            for e in entries:
                row = (_ts(e["start"]), _ts(e.get("stop")), e.get("description") or "",
                       self.projects.get(e.get("project_id") or 0, ""), ", ".join(e.get("tags") or []))
                seen.add(e["id"])
                old = before.get(e["id"])
                if old != row:
                    changed += 1
                    if old is None:
                        if row[1] is not None and row[0] < time.time() - 1800:
                            touched.append((row[0], row[1]))   # an entry added after the fact
                    else:
                        touched += _edit_ranges(old, row, synced_at.get(e["id"]))
                self.conn.execute(
                    """INSERT INTO toggl_entries(id, start, stop, description, project, tags, synced)
                       VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET start=excluded.start, stop=excluded.stop,
                       description=excluded.description, project=excluded.project, tags=excluded.tags, synced=excluded.synced""",
                    (e["id"], *row, time.time()))
            # Local entries inside the fetched window that Toggl no longer returns were deleted there.
            gone = [i for i in before if i not in seen]
            if gone:
                self.conn.executemany("DELETE FROM toggl_entries WHERE id=?", [(i,) for i in gone])
                touched += [(before[i][0], before[i][1] or synced_at.get(i) or before[i][0]) for i in gone]
            queued = queue_reevaluation(self.conn, touched, "Toggl entries corrected")
        if changed or gone:
            log.info("toggl: %d entries in the last %.0fh, %d changed, %d deleted, %d stretch(es) queued for re-evaluation",
                     len(entries), lookback_hours, changed, len(gone), queued)
        return {"fetched": len(entries), "changed": changed, "deleted": len(gone), "hours": lookback_hours,
                "reevaluate": queued}


def _edit_ranges(old: tuple, new: tuple, old_synced: float | None) -> list[tuple[float, float]]:
    """Which stretches an edit to an entry invalidates. An entry simply stopping is NOT an edit:
    nothing judged while it ran assumed anything about when it would end."""
    os_, oe, ns, ne = old[0], old[1], new[0], new[1]
    old_end = oe if oe is not None else (old_synced or os_)   # running: judged only up to the last sync
    out: list[tuple[float, float]] = []
    if old[2:] != new[2:]:                 # description / project / tags: everything it covered, then and now
        out.append((min(os_, ns), max(old_end, ne if ne is not None else time.time())))
        return out
    if abs(os_ - ns) > 1:                  # start moved
        out.append((min(os_, ns), max(os_, ns)))
    if ne is not None and oe is None:      # it stopped; an edit only if the stop is before what we assumed
        if ne < old_end - 1:
            out.append((ne, old_end))
    elif ne is not None and oe is not None and abs(oe - ne) > 1:   # stop moved
        out.append((min(oe, ne), max(oe, ne)))
    elif ne is None and oe is not None:    # restarted / un-stopped
        out.append((oe, time.time()))
    return out


def queue_reevaluation(conn, ranges: list[tuple[float, float]], reason: str, pad: float = 120.0) -> int:
    """Queue already-judged stretches for re-evaluation because their upstream data changed.
    Overlapping ranges are merged; ranges with nothing judged in them are skipped."""
    merged: list[list[float]] = []
    for a, b in sorted((min(a, b) - pad, max(a, b) + pad) for a, b in ranges):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    n = 0
    for a, b in merged:
        judged = conn.execute("""SELECT 1 FROM seg_evals e JOIN segments g ON g.id = e.segment_id
                                 WHERE g.start < ? AND COALESCE(g.end, g.start) > ? LIMIT 1""", (b, a)).fetchone()
        if not judged:
            continue
        dup = conn.execute("SELECT 1 FROM reeval_queue WHERE done_at IS NULL AND t0 <= ? AND t1 >= ?", (a, b)).fetchone()
        if dup:
            continue
        conn.execute("INSERT INTO reeval_queue(t0, t1, reason, created) VALUES (?,?,?,?)", (a, b, reason, time.time()))
        n += 1
    return n


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
