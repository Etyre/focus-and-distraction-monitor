from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .config import DATA_DIR, day_bounds, ensure_dirs, logical_date

DB_PATH = DATA_DIR / "focus.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    start REAL NOT NULL,
    end REAL,
    app TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    domain TEXT NOT NULL DEFAULT '',
    first_screenshot TEXT,
    last_screenshot TEXT,
    idle INTEGER NOT NULL DEFAULT 0,
    neutral INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS segments_start ON segments(start);

CREATE TABLE IF NOT EXISTS switches (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    from_segment INTEGER REFERENCES segments(id),
    to_segment INTEGER NOT NULL REFERENCES segments(id),
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | classified | reviewed | transit
    group_id INTEGER                          -- for transit switches: the terminal switch of the transition
);
CREATE INDEX IF NOT EXISTS switches_ts ON switches(ts);
CREATE INDEX IF NOT EXISTS switches_status ON switches(status);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    switch_id INTEGER NOT NULL REFERENCES switches(id),
    model TEXT NOT NULL,
    p_continuation REAL NOT NULL,
    p_interruption REAL NOT NULL,
    p_distraction REAL NOT NULL,
    p_task_change REAL NOT NULL DEFAULT 0,
    rationale TEXT,
    cost_usd REAL NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    UNIQUE(switch_id, model)
);

CREATE TABLE IF NOT EXISTS ensemble (
    switch_id INTEGER PRIMARY KEY REFERENCES switches(id),
    p_continuation REAL NOT NULL,
    p_interruption REAL NOT NULL,
    p_distraction REAL NOT NULL,
    p_task_change REAL NOT NULL DEFAULT 0,
    label TEXT NOT NULL,
    uncertain INTEGER NOT NULL DEFAULT 0,
    weights TEXT,
    importance REAL NOT NULL DEFAULT 1.0,   -- >1 when the switch leaves an ongoing focus span
    focus_run_min REAL NOT NULL DEFAULT 0,  -- length of the focused run preceding the switch
    created REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS inactivity (   -- short pauses (no input) that did NOT become idle segments
    id INTEGER PRIMARY KEY,
    start REAL NOT NULL,
    end REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS inactivity_start ON inactivity(start);

CREATE TABLE IF NOT EXISTS reviews (
    switch_id INTEGER PRIMARY KEY REFERENCES switches(id),
    label TEXT NOT NULL,
    note TEXT,
    created REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS seg_evals (        -- the chunk evaluator's per-segment judgments
    segment_id INTEGER PRIMARY KEY REFERENCES segments(id),
    content TEXT NOT NULL,        -- creative | task | reading | planning | other | passive
    switch_label TEXT,            -- nature of the switch INTO this segment:
                                  --   continuation | interruption | return
    interruption_kind TEXT,       -- when interruption: self_distraction | focus_start | detour
    uncertain INTEGER NOT NULL DEFAULT 0,  -- the evaluator was genuinely unsure -> flag for review
    rationale TEXT,
    created REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS seg_predictions (  -- per-model probability outputs for each segment
    segment_id INTEGER NOT NULL REFERENCES segments(id),
    model TEXT NOT NULL,          -- heuristic | learned | claude
    p_content TEXT NOT NULL,      -- JSON {label: prob}
    p_switch TEXT NOT NULL,
    p_kind TEXT,
    cost_usd REAL NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    UNIQUE(segment_id, model)
);
CREATE INDEX IF NOT EXISTS seg_predictions_seg ON seg_predictions(segment_id);

CREATE TABLE IF NOT EXISTS audit_runs (       -- once-a-day whole-day LLM error sweep
    day TEXT PRIMARY KEY,
    n_flags INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    created REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS seg_reviews (      -- the user's edits to those judgments (authoritative,
    segment_id INTEGER PRIMARY KEY,           --  and few-shot training signal for the evaluator)
    content TEXT,
    switch_label TEXT,
    interruption_kind TEXT,
    note TEXT,
    created REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (        -- chunk evaluation bookkeeping (spend accounting)
    id INTEGER PRIMARY KEY,
    t0 REAL NOT NULL,
    t1 REAL NOT NULL,
    n_segments INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS eval_runs_t0 ON eval_runs(t0);

CREATE TABLE IF NOT EXISTS thread_calls (     -- small-LLM judgment for ambiguous 'continuation'
    switch_id INTEGER PRIMARY KEY REFERENCES switches(id),  -- switches: WHAT is being continued -
    thread TEXT NOT NULL,                     -- 'work' | 'interruption' | 'distraction'
    rationale TEXT,
    cost_usd REAL NOT NULL DEFAULT 0,
    created REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS span_reviews (     -- the user's verdict on each hypothesized span
    id INTEGER PRIMARY KEY,
    start REAL NOT NULL,
    end REAL NOT NULL,
    verdict TEXT NOT NULL,        -- 'focus' | 'not_focus'
    subtype TEXT,                 -- confirmed/changed subtype when verdict='focus'
    note TEXT,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS span_reviews_start ON span_reviews(start);

CREATE TABLE IF NOT EXISTS span_summaries (   -- Claude's summary of each finished (hypothesized) span
    id INTEGER PRIMARY KEY,
    start REAL NOT NULL,
    end REAL NOT NULL,
    summary TEXT NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS span_summaries_start ON span_summaries(start);
"""


def connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(segments)")}
    if "neutral" not in cols:  # migrate older databases
        conn.execute("ALTER TABLE segments ADD COLUMN neutral INTEGER NOT NULL DEFAULT 0")
    scols = {r[1] for r in conn.execute("PRAGMA table_info(switches)")}
    if "group_id" not in scols:
        conn.execute("ALTER TABLE switches ADD COLUMN group_id INTEGER")
    ecols = {r[1] for r in conn.execute("PRAGMA table_info(ensemble)")}
    if "importance" not in ecols:
        conn.execute("ALTER TABLE ensemble ADD COLUMN importance REAL NOT NULL DEFAULT 1.0")
        conn.execute("ALTER TABLE ensemble ADD COLUMN focus_run_min REAL NOT NULL DEFAULT 0")
    if "activity" not in ecols and "importance" in ecols:
        conn.execute("ALTER TABLE ensemble ADD COLUMN activity TEXT")
    elif "activity" not in {r[1] for r in conn.execute("PRAGMA table_info(ensemble)")}:
        conn.execute("ALTER TABLE ensemble ADD COLUMN activity TEXT")
    vcols = {r[1] for r in conn.execute("PRAGMA table_info(seg_evals)")}
    if vcols and "uncertain" not in vcols:
        conn.execute("ALTER TABLE seg_evals ADD COLUMN uncertain INTEGER NOT NULL DEFAULT 0")
    return conn


@contextmanager
def tx(conn: sqlite3.Connection):
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


# ----- convenience queries -----------------------------------------------------------------

def segment(conn, seg_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM segments WHERE id=?", (seg_id,)).fetchone()


def segments_between(conn, t0: float, t1: float) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM segments WHERE start < ? AND COALESCE(end, ?) > ? ORDER BY start",
        (t1, time.time(), t0),
    ).fetchall()


def segments_before(conn, ts: float, n: int) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT * FROM segments WHERE start < ? ORDER BY start DESC LIMIT ?", (ts, n)
    ).fetchall()
    return list(reversed(rows))


def segments_after(conn, ts: float, n: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM segments WHERE start > ? ORDER BY start LIMIT ?", (ts, n)
    ).fetchall()


def switch_full(conn, switch_id: int) -> dict | None:
    sw = conn.execute("SELECT * FROM switches WHERE id=?", (switch_id,)).fetchone()
    if not sw:
        return None
    return {
        "switch": dict(sw),
        "from": dict(segment(conn, sw["from_segment"])) if sw["from_segment"] else None,
        "to": dict(segment(conn, sw["to_segment"])),
        "predictions": [dict(r) for r in conn.execute(
            "SELECT * FROM predictions WHERE switch_id=?", (switch_id,))],
        "ensemble": (lambda r: dict(r) if r else None)(
            conn.execute("SELECT * FROM ensemble WHERE switch_id=?", (switch_id,)).fetchone()),
        "review": (lambda r: dict(r) if r else None)(
            conn.execute("SELECT * FROM reviews WHERE switch_id=?", (switch_id,)).fetchone()),
    }


def reviews_with_context(conn, limit: int | None = None) -> list[dict]:
    q = """SELECT r.switch_id, r.label, r.note, r.created, s.ts,
                  f.app AS from_app, f.title AS from_title, f.url AS from_url, f.domain AS from_domain,
                  f.start AS from_start, f.end AS from_end,
                  t.app AS to_app, t.title AS to_title, t.url AS to_url, t.domain AS to_domain,
                  t.start AS to_start, t.end AS to_end
           FROM reviews r JOIN switches s ON s.id=r.switch_id
           LEFT JOIN segments f ON f.id=s.from_segment
           JOIN segments t ON t.id=s.to_segment
           ORDER BY r.created DESC"""
    if limit:
        q += f" LIMIT {int(limit)}"
    return [dict(r) for r in conn.execute(q)]


def spend_today(conn) -> float:
    start = day_bounds(logical_date())[0]
    row = conn.execute("SELECT COALESCE(SUM(cost_usd),0) FROM predictions WHERE created >= ?",
                       (start,)).fetchone()
    srow = conn.execute("SELECT COALESCE(SUM(cost_usd),0) FROM span_summaries WHERE created >= ?",
                        (start,)).fetchone()
    trow = conn.execute("SELECT COALESCE(SUM(cost_usd),0) FROM thread_calls WHERE created >= ?",
                        (start,)).fetchone()
    erow = conn.execute("SELECT COALESCE(SUM(cost_usd),0) FROM eval_runs WHERE created >= ?",
                        (start,)).fetchone()
    return float(row[0]) + float(srow[0]) + float(trow[0]) + float(erow[0])
