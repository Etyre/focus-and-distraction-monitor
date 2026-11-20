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
        if not self.projects_at:
            # The project list has never been fetched (it failed just now while the entries call
            # worked): writing every entry with an empty project would read as a correction to
            # each one, and queue days of re-evaluation - twice, as the names came back.
            log.warning("toggl: entries fetched but no project list yet - not storing this round")
            return None
        changed = 0
        touched: list[tuple[float, float]] = []   # time ranges whose intent statement changed
        with self.conn:
            before = {r["id"]: (r["start"], r["stop"], r["description"], r["project"], r["tags"]) for r in
                      self.conn.execute("SELECT * FROM toggl_entries WHERE start >= ?", (t0.timestamp(),))}
            synced_at = {r["id"]: r["synced"] for r in
                         self.conn.execute("SELECT id, synced FROM toggl_entries WHERE stop IS NULL")}
            # An entry with no stop is taken by the evaluator to cover everything up to the
            # moment it judges, however long ago the last successful sync was. So the 'old end'
            # of a running entry is the end of the newest judged segment, not the last sync:
            # with sync failing from 00:01 to 02:21 on 2026-11-19, an entry that had stopped at
            # 00:24 was shown as running in every call up to 02:14, and the last-sync rule
            # queued nothing after 00:26 (segments 36083-36088 kept judgments made under it).
            judged_to = self.conn.execute("""SELECT MAX(COALESCE(g.end, g.start)) FROM seg_evals e
                                             JOIN segments g ON g.id = e.segment_id""").fetchone()[0] or 0.0
            seen = set()
            details: list[dict] = []   # what changed, for the note the evaluator gets (queue_reevaluation)
            for e in entries:
                old = before.get(e["id"])
                if old is None:   # an entry that began before the fetched window is not new
                    r0 = self.conn.execute("SELECT * FROM toggl_entries WHERE id=?", (e["id"],)).fetchone()
                    old = (r0["start"], r0["stop"], r0["description"], r0["project"], r0["tags"]) if r0 is not None else None
                pid = e.get("project_id") or 0
                # a project newer than the cached list (refreshed every 6 h): keep the name on record
                project = self.projects.get(pid, old[3] if (old and pid) else "") if pid else ""
                row = (_ts(e["start"]), _ts(e.get("stop")), e.get("description") or "", project, ", ".join(e.get("tags") or []))
                seen.add(e["id"])
                if old != row:
                    changed += 1
                    if old is None:
                        # An entry the app has not seen before. Whatever was already judged in
                        # its time was judged without it; queue_reevaluation skips a range with
                        # nothing judged in it, which is the normal case of a timer just started.
                        # (Until 2026-11-20 this needed the entry to start 30+ minutes ago, while
                        # judging happens at 15: segments 28906-28911 on 11-03 kept judgments
                        # made before the entry for that meeting was added.)
                        rng = [(row[0], row[1] if row[1] is not None else max(judged_to, row[0]))]
                    else:
                        rng = _edit_ranges(old, row, synced_at.get(e["id"]), judged_to)
                    touched += rng
                    details += [{"a": a, "b": b, "old": old, "new": row} for a, b in rng]
                self.conn.execute(
                    """INSERT INTO toggl_entries(id, start, stop, description, project, tags, synced)
                       VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET start=excluded.start, stop=excluded.stop,
                       description=excluded.description, project=excluded.project, tags=excluded.tags, synced=excluded.synced""",
                    (e["id"], *row, time.time()))
            # Local entries inside the fetched window that Toggl no longer returns were deleted there.
            gone = [i for i in before if i not in seen]
            if gone:
                self.conn.executemany("DELETE FROM toggl_entries WHERE id=?", [(i,) for i in gone])
                for i in gone:
                    rng = (before[i][0], before[i][1] or max(synced_at.get(i) or 0.0, judged_to, before[i][0]))
                    touched.append(rng)
                    details.append({"a": rng[0], "b": rng[1], "old": before[i], "new": None})
            queued = queue_reevaluation(self.conn, touched, "Toggl entries corrected", details=details)
        if changed or gone:
            log.info("toggl: %d entries in the last %.0fh, %d changed, %d deleted, %d stretch(es) queued for re-evaluation",
                     len(entries), lookback_hours, changed, len(gone), queued)
        return {"fetched": len(entries), "changed": changed, "deleted": len(gone), "hours": lookback_hours,
                "reevaluate": queued}


def _edit_ranges(old: tuple, new: tuple, old_synced: float | None,
                 judged_to: float = 0.0) -> list[tuple[float, float]]:
    """Which stretches an edit to an entry invalidates. An entry stopping at about the time we
    last saw it running is NOT an edit. One found to have stopped earlier than the newest
    judgment is: everything judged after its real stop was judged with it shown as running
    (`judged_to` = end of the newest judged segment; see resync)."""
    os_, oe, ns, ne = old[0], old[1], new[0], new[1]
    old_end = oe if oe is not None else max(old_synced or os_, judged_to)   # running: shown as covering all judged time
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


def queue_reevaluation(conn, ranges: list[tuple[float, float]], reason: str, pad: float = 120.0,
                       details: list[dict] | None = None) -> int:
    """Queue already-judged stretches for re-evaluation because their upstream data changed.
    Overlapping ranges are merged; ranges with nothing judged in them are skipped. `details`
    ([{a, b, old, new}], old / new = (start, stop, description, project, tags) or None) says what
    changed; it is kept beside the queue row (kv "reeval.detail.<row id>", JSON) so the note to
    the evaluator can say what the entries were when it first judged (the table itself keeps
    only the corrected entry)."""
    def keep(row_id, a, b):
        mine = [d for d in (details or []) if min(d["a"], d["b"]) - pad < b and max(d["a"], d["b"]) + pad > a]
        if not mine:
            return
        try:
            r = conn.execute("SELECT value FROM kv WHERE key=?", (f"reeval.detail.{row_id}",)).fetchone()
            had = json.loads(r[0]) if r and r[0] else []
            conn.execute("INSERT OR REPLACE INTO kv(key, value) VALUES (?,?)",
                         (f"reeval.detail.{row_id}", json.dumps((had + [{"old": d["old"], "new": d["new"], "at": time.time()} for d in mine])[-40:])))
        except Exception:   # the note is extra; never lose the queue row for it
            log.exception("could not record what changed for re-evaluation %s", row_id)
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
        dup = conn.execute("SELECT id FROM reeval_queue WHERE done_at IS NULL AND t0 <= ? AND t1 >= ?", (a, b)).fetchone()
        if dup:
            keep(dup[0], a, b)
            continue
        keep(conn.execute("INSERT INTO reeval_queue(t0, t1, reason, created) VALUES (?,?,?,?)",
                          (a, b, reason, time.time())).lastrowid, a, b)
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
