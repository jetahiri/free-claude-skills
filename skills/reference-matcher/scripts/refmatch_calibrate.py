"""Calibration for the reference matcher: your verdicts sit next to the scores, and your eye wins.

Called through refmatch.py (verdict, agreement). When a verdict disagrees with the tool's pass or fail, the job kind's
bar moves just far enough that the run would have been judged your way, and the move is logged in refmatch_calibration
with the date, your words and the scores. Once a kind has 8 or more verdicts of both kinds, the weights are refit to them
(logistic regression on the 0..1 scores) and the combined bar moves to the cut that agrees with the most verdicts.
Agreement is measured against the tool's answer as it stood when it scored, so the report shows the tool getting better
(or not) over time, week by week.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refmatch  # noqa: E402
import refmatch_store  # noqa: E402
from refmatch import METRICS, RefmatchError, load_bars, save_bars  # noqa: E402

MIN_VERDICTS_TO_REFIT = 8
# blind evaluation (the verdict came before the score was seen) cannot be anchored by it, so it counts most in the refit
SEEN_WEIGHT = {"blind": 3.0, "seen-possibly": 1.5, "seen": 1.0, None: 1.0}


def _db():
    return refmatch_store.session(refmatch.BARS_FILE)


def _log(con, kind: str, match_id, change: str, before, after, reason: str) -> None:
    con.execute("INSERT INTO refmatch_calibration (created_at, kind, match_id, change, before, after, reason)"
                " VALUES (?,?,?,?,?,?,?)", (datetime.now().isoformat(timespec="seconds"), kind, match_id, change,
                                           json.dumps(before), json.dumps(after), reason))


def _verdict_rows(con, kind: str) -> list[dict]:
    rows = con.execute("SELECT id, metrics, verdict, verdict_seen FROM reference_matches WHERE kind=?"
                       " AND verdict IS NOT NULL AND combined IS NOT NULL ORDER BY verdict_at, id", (kind,)).fetchall()
    return [{"id": r[0], "norm": json.loads(r[1] or "{}").get("norm", {}), "accept": r[2] == "accept",
             "w": SEEN_WEIGHT.get(r[3], 1.0)} for r in rows]


def _best_cut(scores: list, rows: list, latest_id: int) -> float:
    """The combined bar that agrees with the most (weighted) verdicts; ties go to the cut that agrees with the latest
    verdict, then to the highest cut (the stricter bar)."""
    cuts = sorted({round(s, 1) for s in scores} | {round(s + 0.1, 1) for s in scores})

    def key(c):
        agree = sum(r["w"] * ((sc >= c) == r["accept"]) for sc, r in zip(scores, rows))
        latest = next(((sc >= c) == r["accept"] for sc, r in zip(scores, rows) if r["id"] == latest_id), True)
        return (round(agree, 6), latest, c)
    return float(max(cuts, key=key))


def _set_combined_bar(con, kind: str, bar: dict, match_id: int, quote: str) -> list:
    """Move the combined bar the least that agrees with every verdict on record; when no bar can, take the cut that
    agrees with the most (blind verdicts weigh most) and log the conflict."""
    rows = _verdict_rows(con, kind)
    scores = [refmatch.combined_score(r["norm"], bar["weights"]) or 0.0 for r in rows]
    acc = [s for s, r in zip(scores, rows) if r["accept"]]
    rej = [s for s, r in zip(scores, rows) if not r["accept"]]
    cur = bar["combined_min"]
    lo = max(rej) if rej else None          # the bar must be above every rejected score
    hi = min(acc) if acc else None          # and at or below every accepted score
    if lo is not None and hi is not None and lo >= hi:
        new = _best_cut(scores, rows, match_id)
        change, reason = "conflict", (f"after match #{match_id}{quote}: a rejected run scored {lo} and an accepted run "
                                      f"{hi}, so no bar agrees with every verdict; bar set to the cut that agrees with "
                                      "the most (blind verdicts weigh most) and the weights need refitting")
    elif (lo is None or cur > lo) and (hi is None or cur <= hi):
        return []
    else:
        if lo is not None and hi is not None:
            new = round((lo + hi) / 2, 1)
            new = new if lo < new <= hi else hi
        elif lo is not None:
            new = round(lo + 1, 1)
        else:
            new = round(hi - 0.5, 1)
        change, reason = "combined bar moved", f"match #{match_id}{quote}: bar moved so every verdict on record agrees"
    if new == cur:
        if change == "conflict":
            _log(con, kind, match_id, change, cur, cur, reason)
            return ["conflict logged; bar unchanged"]
        return []
    bar["combined_min"] = new
    _log(con, kind, match_id, change, cur, new, reason)
    return [f"{change}: {cur} -> {new}"]


def _clashes(con, kind: str, bar: dict, match_id: int) -> list:
    """Replay every verdict on record against the bars now in force; log each one the tool would judge the other way."""
    out = []
    for r in _verdict_rows(con, kind):
        _, passed, fails = refmatch.judge(r["norm"], bar)
        if passed is not None and (passed == 1) != r["accept"]:
            out.append(f"#{r['id']} ({'accepted' if r['accept'] else 'rejected'}, tool now says "
                       f"{'pass' if passed else 'fail'}{': ' + '; '.join(fails) if fails else ''})")
    if out:
        _log(con, kind, match_id, "clash", None, out, "verdicts on record that the bars in force now judge the other way")
    return out


def _refit_weights(con, kind: str, bar: dict, match_id: int) -> bool:
    rows = _verdict_rows(con, kind)
    ys = [r["accept"] for r in rows]
    if len(rows) < MIN_VERDICTS_TO_REFIT or all(ys) or not any(ys):
        return False
    from sklearn.linear_model import LogisticRegression
    keys = [k for k in METRICS if bar["weights"].get(k, 0) > 0]
    # a missing score enters the fit at the mean of the run's other scores, as the combined score leaving it out does
    X = np.array([[r["norm"][k] if r["norm"].get(k) is not None else
                   np.mean([v for v in r["norm"].values() if v is not None] or [0]) for k in keys] for r in rows])
    coef = LogisticRegression(C=1.0).fit(X, ys, sample_weight=[r["w"] for r in rows]).coef_[0].clip(0.02, None)
    weights = {k: round(float(c / coef.sum()), 3) for k, c in zip(keys, coef)}
    scores = [refmatch.combined_score(r["norm"], weights) or 0.0 for r in rows]
    before = {"weights": dict(bar["weights"]), "combined_min": bar["combined_min"]}
    bar["weights"].update(weights)
    bar["combined_min"] = _best_cut(scores, rows, match_id)
    _log(con, kind, match_id, "weights refit", before, {"weights": bar["weights"], "combined_min": bar["combined_min"]},
         f"{len(rows)} verdicts on record ({sum(ys)} accepted): weights refit to your verdicts")
    return True


def verdict(match_id: int, decision: str, words: str = "", seen: str = "seen") -> dict:
    """Record your verdict on a scored recreation and make the kind's bars agree with every verdict on record (your
    eye wins); any verdict the bars still judge the other way is logged as a clash. A second verdict on the same run
    replaces the first, and the first is kept in the run's notes."""
    if decision not in ("accept", "reject"):
        raise RefmatchError("verdict is accept or reject")
    if seen not in ("blind", "seen", "seen-possibly"):
        raise RefmatchError("seen is blind, seen or seen-possibly")
    cfg = load_bars()
    changes: list = []
    clashes: list = []
    with _db() as con:
        row = con.execute("SELECT kind, combined, metrics, passed, notes, verdict, verdict_words, verdict_at, verdict_seen"
                          " FROM reference_matches WHERE id=?", (match_id,)).fetchone()
        if row is None:
            raise RefmatchError(f"no reference match #{match_id}")
        kind, combined, metrics, passed = row[0], row[1], json.loads(row[2] or "{}"), row[3]
        notes = json.loads(row[4] or "[]")
        if row[5]:
            notes.append(f'earlier verdict replaced: {row[5]} "{row[6] or ""}" at {row[7]} ({row[8] or "seen"})')
        con.execute("UPDATE reference_matches SET verdict=?, verdict_words=?, verdict_at=?, verdict_seen=?, notes=?"
                    " WHERE id=?", (decision, words, datetime.now().isoformat(timespec="seconds"), seen,
                                    json.dumps(notes), match_id))
        bar = cfg["kinds"].get(kind)
        quote = (f' ("{words}")' if words else "") + f", {seen}"
        if bar is not None and combined is not None:
            norm = metrics.get("norm", {})
            if decision == "accept":
                lowered = {k: round(norm[k] - 0.01, 3) for k, v in bar["min"].items()
                           if norm.get(k) is not None and norm[k] < v}
                if lowered:
                    before = dict(bar["min"])
                    bar["min"].update(lowered)
                    _log(con, kind, match_id, "minimums lowered", before, bar["min"],
                         f"accepted match #{match_id}{quote}, which was under these minimums")
                    changes.append(f"minimums lowered: {lowered}")
            if _refit_weights(con, kind, bar, match_id):
                changes.append("weights refit")
            else:
                changes += _set_combined_bar(con, kind, bar, match_id, quote)
            clashes = _clashes(con, kind, bar, match_id)
        if changes:
            save_bars(cfg)
    agrees = None if passed is None else (passed == 1) == (decision == "accept")
    return {"match_id": match_id, "verdict": decision, "tool_said": {1: "pass", 0: "fail", None: "incomplete"}[passed],
            "agrees": agrees, "changes": changes, "clashes": clashes}


def agreement(kind: str | None = None) -> dict:
    """How often the tool's pass or fail (as it stood when it scored) matched your verdict, per kind and per week."""
    sql = ("SELECT kind, strftime('%Y-%W', verdict_at) wk, passed, verdict FROM reference_matches"
           " WHERE verdict IS NOT NULL" + (" AND kind=?" if kind else ""))
    out: dict = {}
    with _db() as con:
        rows = con.execute(sql, [kind] if kind else []).fetchall()
    for r in rows:
        k = out.setdefault(r["kind"], {"verdicts": 0, "scored": 0, "agree": 0, "weeks": {}})
        k["verdicts"] += 1
        if r["passed"] is None:
            continue
        ok = (r["passed"] == 1) == (r["verdict"] == "accept")
        wk = k["weeks"].setdefault(r["wk"], {"scored": 0, "agree": 0})
        for d in (k, wk):
            d["scored"] += 1
            d["agree"] += ok
    for k in out.values():
        k["agreement"] = round(k["agree"] / k["scored"], 3) if k["scored"] else None
    return out


def verdict_without_image(kind: str, reference: str, label: str, decision: str, words: str,
                          build: int | None = None, seen: str = "blind") -> int:
    """A verdict on a recreation whose image was not kept: on the record, never used to tune (no scores)."""
    now = datetime.now().isoformat(timespec="seconds")
    with _db() as con:
        return con.execute(
            "INSERT INTO reference_matches (created_at, kind, build_id, label, reference, notes, verdict, verdict_words,"
            " verdict_at, verdict_seen) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (now, kind, build, label, reference, json.dumps(["image not kept; verdict recorded without scores"]),
             decision, words, now, seen)).lastrowid
