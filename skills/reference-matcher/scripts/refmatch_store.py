"""Local record for the reference matcher: one SQLite file, created on first use.

reference_matches holds one row per scored recreation (the tool's pass or fail at the time and, once you have looked,
your verdict and words); refmatch_calibration logs every change to a job kind's pass bar or weights, with the reason.
passed: 1 pass, 0 fail, NULL incomplete (a required score could not be computed) or no image kept.

The file sits beside refmatch_bars.json as refmatch_records.sqlite; set REFMATCH_DB to put it elsewhere.
"""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS reference_matches (
    id            INTEGER PRIMARY KEY,
    created_at    TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    build_id      INTEGER,
    label         TEXT,
    reference     TEXT    NOT NULL,
    render        TEXT,
    report_dir    TEXT,
    combined      REAL,
    metrics       TEXT    NOT NULL DEFAULT '{}',
    bar           TEXT    NOT NULL DEFAULT '{}',
    passed        INTEGER,
    notes         TEXT    NOT NULL DEFAULT '[]',
    verdict       TEXT    CHECK (verdict IN ('accept', 'reject')),
    verdict_words TEXT,
    verdict_at    TEXT,
    verdict_seen  TEXT    CHECK (verdict_seen IN ('blind', 'seen', 'seen-possibly'))
);
CREATE INDEX IF NOT EXISTS idx_reference_matches_kind ON reference_matches(kind, created_at);
CREATE TABLE IF NOT EXISTS refmatch_calibration (
    id          INTEGER PRIMARY KEY,
    created_at  TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    match_id    INTEGER REFERENCES reference_matches(id),
    change      TEXT    NOT NULL,
    before      TEXT    NOT NULL,
    after       TEXT    NOT NULL,
    reason      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_refmatch_calibration_kind ON refmatch_calibration(kind, created_at);
"""


def db_path(bars_file: Path) -> Path:
    return Path(os.environ.get("REFMATCH_DB") or Path(bars_file).with_name("refmatch_records.sqlite"))


@contextmanager
def session(bars_file: Path):
    """A connection that creates the tables on first use, commits on success and always closes."""
    con = sqlite3.connect(db_path(bars_file))
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        with con:
            yield con
    finally:
        con.close()
