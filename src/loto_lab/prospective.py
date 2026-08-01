from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime, time
from pathlib import Path
from statistics import fmean
from zoneinfo import ZoneInfo

from . import __version__
from .domain import Draw
from .probability import rank_probabilities, total_outcomes
from .value import ValueReport, _lower_rank_ev, _poisson_share_factor
from .value_backtest import _draw_jackpot, _draw_ticket_count

GENESIS_HASH = "0" * 64
PARIS_TIMEZONE = ZoneInfo("Europe/Paris")
LOTO_RECORDING_CUTOFF = time(20, 15)

LEDGER_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS ledger_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS value_forecasts (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    game TEXT NOT NULL,
    target_date TEXT NOT NULL,
    training_last_date TEXT NOT NULL,
    jackpot REAL NOT NULL,
    estimated_ev REAL NOT NULL,
    ev_ci_low REAL NOT NULL,
    ev_ci_high REAL NOT NULL,
    ticket_price REAL NOT NULL,
    decision TEXT NOT NULL,
    report_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL UNIQUE,
    UNIQUE (game, target_date),
    CHECK (target_date > training_last_date)
);

CREATE TABLE IF NOT EXISTS value_scores (
    id INTEGER PRIMARY KEY,
    forecast_id INTEGER NOT NULL UNIQUE REFERENCES value_forecasts(id),
    scored_at TEXT NOT NULL,
    observed_schedule_ev REAL NOT NULL,
    error REAL NOT NULL,
    absolute_error REAL NOT NULL,
    covered INTEGER NOT NULL CHECK (covered IN (0, 1)),
    observed_positive INTEGER NOT NULL CHECK (observed_positive IN (0, 1)),
    false_positive INTEGER NOT NULL CHECK (false_positive IN (0, 1)),
    false_negative INTEGER NOT NULL CHECK (false_negative IN (0, 1)),
    draw_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL UNIQUE
);

CREATE TRIGGER IF NOT EXISTS value_forecasts_no_update
BEFORE UPDATE ON value_forecasts BEGIN
    SELECT RAISE(ABORT, 'value_forecasts is append-only');
END;

CREATE TRIGGER IF NOT EXISTS value_forecasts_no_delete
BEFORE DELETE ON value_forecasts BEGIN
    SELECT RAISE(ABORT, 'value_forecasts is append-only');
END;

CREATE TRIGGER IF NOT EXISTS value_scores_no_update
BEFORE UPDATE ON value_scores BEGIN
    SELECT RAISE(ABORT, 'value_scores is append-only');
END;

CREATE TRIGGER IF NOT EXISTS value_scores_no_delete
BEFORE DELETE ON value_scores BEGIN
    SELECT RAISE(ABORT, 'value_scores is append-only');
END;
"""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _connect(path: str | Path) -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(LEDGER_SCHEMA)
    connection.execute(
        "INSERT OR IGNORE INTO ledger_metadata(key, value) VALUES ('schema_version', '1')"
    )
    connection.commit()
    return connection


def _forecast_hash(
    *,
    created_at: str,
    model_version: str,
    game: str,
    target_date: str,
    training_last_date: str,
    jackpot: float,
    estimated_ev: float,
    ev_ci_low: float,
    ev_ci_high: float,
    ticket_price: float,
    decision: str,
    report_json: str,
    previous_hash: str,
) -> str:
    return _digest(
        {
            "created_at": created_at,
            "model_version": model_version,
            "game": game,
            "target_date": target_date,
            "training_last_date": training_last_date,
            "jackpot": float(jackpot),
            "estimated_ev": float(estimated_ev),
            "ev_ci_low": float(ev_ci_low),
            "ev_ci_high": float(ev_ci_high),
            "ticket_price": float(ticket_price),
            "decision": decision,
            "previous_hash": previous_hash,
            "report_json": report_json,
        }
    )


def _score_hash(
    scored_at: str,
    forecast_hash: str,
    previous_hash: str,
    draw_json: str,
    metrics: dict[str, float | int],
) -> str:
    return _digest(
        {
            "scored_at": scored_at,
            "forecast_hash": forecast_hash,
            "previous_hash": previous_hash,
            "draw_json": draw_json,
            "metrics": metrics,
        }
    )


def _recording_is_open(target_date: date, current_time: datetime) -> bool:
    local_time = current_time.astimezone(PARIS_TIMEZONE)
    if target_date > local_time.date():
        return True
    if target_date < local_time.date():
        return False
    cutoff = datetime.combine(target_date, LOTO_RECORDING_CUTOFF, PARIS_TIMEZONE)
    return local_time < cutoff


def record_value_forecast(
    ledger: str | Path,
    report: ValueReport,
    *,
    model_version: str = __version__,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    if report.target_date is None:
        raise ValueError("Une prevision prospective exige une date cible")
    if report.target_date <= report.training_last_date:
        raise ValueError("La date cible doit etre posterieure aux donnees d'apprentissage")
    current_time = datetime.now(UTC)
    if not _recording_is_open(report.target_date, current_time):
        raise ValueError("Une prevision prospective ne peut pas etre retroactive")
    created_at = current_time.astimezone(UTC).isoformat()
    report_json = _canonical_json(
        {"report": report.to_dict(), "provenance": provenance or {}}
    )
    connection = _connect(ledger)
    try:
        connection.execute("BEGIN IMMEDIATE")
        previous = connection.execute(
            "SELECT record_hash FROM value_forecasts ORDER BY id DESC LIMIT 1"
        ).fetchone()
        previous_hash = str(previous["record_hash"]) if previous else GENESIS_HASH
        record_hash = _forecast_hash(
            created_at=created_at,
            model_version=model_version,
            game=report.game,
            target_date=report.target_date.isoformat(),
            training_last_date=report.training_last_date.isoformat(),
            jackpot=report.jackpot,
            estimated_ev=report.estimated_ev,
            ev_ci_low=report.ev_ci_low,
            ev_ci_high=report.ev_ci_high,
            ticket_price=report.ticket_price,
            decision=report.decision,
            report_json=report_json,
            previous_hash=previous_hash,
        )
        try:
            cursor = connection.execute(
                """
                INSERT INTO value_forecasts(
                    created_at, model_version, game, target_date, training_last_date,
                    jackpot, estimated_ev, ev_ci_low, ev_ci_high, ticket_price, decision,
                    report_json, previous_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    model_version,
                    report.game,
                    report.target_date.isoformat(),
                    report.training_last_date.isoformat(),
                    report.jackpot,
                    report.estimated_ev,
                    report.ev_ci_low,
                    report.ev_ci_high,
                    report.ticket_price,
                    report.decision,
                    report_json,
                    previous_hash,
                    record_hash,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"Une prevision {report.game} existe deja pour {report.target_date}"
            ) from error
        connection.commit()
        return {
            "forecast_id": int(cursor.lastrowid),
            "created_at": created_at,
            "game": report.game,
            "target_date": report.target_date.isoformat(),
            "training_last_date": report.training_last_date.isoformat(),
            "model_version": model_version,
            "decision": report.decision,
            "estimated_ev": report.estimated_ev,
            "ev_ci_low": report.ev_ci_low,
            "ev_ci_high": report.ev_ci_high,
            "previous_hash": previous_hash,
            "record_hash": record_hash,
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _draw_payload(draw: Draw) -> dict[str, object]:
    return {
        "game": draw.game,
        "draw_date": draw.draw_date,
        "main": draw.main,
        "chance": draw.chance,
        "prizes": [
            {"rank": prize.rank, "winners": prize.winners, "payout": prize.payout}
            for prize in draw.prizes
        ],
        "code_winners": draw.code_winners,
        "code_payout": draw.code_payout,
    }


def _observed_schedule_ev(draw: Draw) -> float | None:
    probabilities = {item.rank: item.probability for item in rank_probabilities()}
    jackpot = _draw_jackpot(draw)
    actual_tickets = _draw_ticket_count(draw, probabilities[9])
    if (
        jackpot is None
        or actual_tickets is None
        or draw.code_winners is None
        or draw.code_payout is None
    ):
        return None
    return (
        _lower_rank_ev(draw, probabilities)
        + draw.code_winners * draw.code_payout / actual_tickets
        + probabilities[1]
        * jackpot
        * _poisson_share_factor(actual_tickets / total_outcomes())
    )


def score_pending_forecasts(
    ledger: str | Path,
    draws: list[Draw],
) -> dict[str, object]:
    by_target = {
        (draw.game, draw.draw_date.isoformat()): draw
        for draw in draws
        if draw.draw_date is not None
    }
    scored_at = datetime.now(UTC).isoformat()
    connection = _connect(ledger)
    scored = []
    skipped = []
    try:
        connection.execute("BEGIN IMMEDIATE")
        pending = connection.execute(
            """
            SELECT f.* FROM value_forecasts f
            LEFT JOIN value_scores s ON s.forecast_id = f.id
            WHERE s.forecast_id IS NULL ORDER BY f.id
            """
        ).fetchall()
        previous = connection.execute(
            "SELECT record_hash FROM value_scores ORDER BY id DESC LIMIT 1"
        ).fetchone()
        previous_hash = str(previous["record_hash"]) if previous else GENESIS_HASH
        for forecast in pending:
            key = (str(forecast["game"]), str(forecast["target_date"]))
            draw = by_target.get(key)
            if draw is None:
                skipped.append({"forecast_id": int(forecast["id"]), "reason": "tirage_absent"})
                continue
            observed_ev = _observed_schedule_ev(draw)
            if observed_ev is None:
                skipped.append(
                    {"forecast_id": int(forecast["id"]), "reason": "bareme_incomplet"}
                )
                continue
            estimated_ev = float(forecast["estimated_ev"])
            ticket_price = float(forecast["ticket_price"])
            eligible = str(forecast["decision"]) == "eligible"
            observed_positive = observed_ev > ticket_price
            metrics: dict[str, float | int] = {
                "observed_schedule_ev": observed_ev,
                "error": estimated_ev - observed_ev,
                "absolute_error": abs(estimated_ev - observed_ev),
                "covered": int(
                    float(forecast["ev_ci_low"])
                    <= observed_ev
                    <= float(forecast["ev_ci_high"])
                ),
                "observed_positive": int(observed_positive),
                "false_positive": int(eligible and not observed_positive),
                "false_negative": int(not eligible and observed_positive),
            }
            draw_json = _canonical_json(_draw_payload(draw))
            record_hash = _score_hash(
                scored_at,
                str(forecast["record_hash"]),
                previous_hash,
                draw_json,
                metrics,
            )
            cursor = connection.execute(
                """
                INSERT INTO value_scores(
                    forecast_id, scored_at, observed_schedule_ev, error, absolute_error,
                    covered, observed_positive, false_positive, false_negative, draw_json,
                    previous_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(forecast["id"]),
                    scored_at,
                    metrics["observed_schedule_ev"],
                    metrics["error"],
                    metrics["absolute_error"],
                    metrics["covered"],
                    metrics["observed_positive"],
                    metrics["false_positive"],
                    metrics["false_negative"],
                    draw_json,
                    previous_hash,
                    record_hash,
                ),
            )
            scored.append(
                {
                    "score_id": int(cursor.lastrowid),
                    "forecast_id": int(forecast["id"]),
                    **metrics,
                    "record_hash": record_hash,
                }
            )
            previous_hash = record_hash
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {"new_scores": len(scored), "scores": scored, "skipped": skipped}


def verify_ledger(ledger: str | Path) -> dict[str, object]:
    connection = _connect(ledger)
    errors = []
    try:
        previous_hash = GENESIS_HASH
        forecasts = connection.execute(
            "SELECT * FROM value_forecasts ORDER BY id"
        ).fetchall()
        forecast_hashes = {}
        for row in forecasts:
            expected = _forecast_hash(
                created_at=str(row["created_at"]),
                model_version=str(row["model_version"]),
                game=str(row["game"]),
                target_date=str(row["target_date"]),
                training_last_date=str(row["training_last_date"]),
                jackpot=float(row["jackpot"]),
                estimated_ev=float(row["estimated_ev"]),
                ev_ci_low=float(row["ev_ci_low"]),
                ev_ci_high=float(row["ev_ci_high"]),
                ticket_price=float(row["ticket_price"]),
                decision=str(row["decision"]),
                report_json=str(row["report_json"]),
                previous_hash=previous_hash,
            )
            if str(row["previous_hash"]) != previous_hash or str(row["record_hash"]) != expected:
                errors.append(f"forecast:{row['id']}")
            previous_hash = str(row["record_hash"])
            forecast_hashes[int(row["id"])] = str(row["record_hash"])
        forecast_head = previous_hash

        previous_hash = GENESIS_HASH
        scores = connection.execute("SELECT * FROM value_scores ORDER BY id").fetchall()
        for row in scores:
            metrics: dict[str, float | int] = {
                "observed_schedule_ev": float(row["observed_schedule_ev"]),
                "error": float(row["error"]),
                "absolute_error": float(row["absolute_error"]),
                "covered": int(row["covered"]),
                "observed_positive": int(row["observed_positive"]),
                "false_positive": int(row["false_positive"]),
                "false_negative": int(row["false_negative"]),
            }
            expected = _score_hash(
                str(row["scored_at"]),
                forecast_hashes.get(int(row["forecast_id"]), "missing"),
                previous_hash,
                str(row["draw_json"]),
                metrics,
            )
            if str(row["previous_hash"]) != previous_hash or str(row["record_hash"]) != expected:
                errors.append(f"score:{row['id']}")
            previous_hash = str(row["record_hash"])
        return {
            "valid": not errors,
            "errors": errors,
            "forecasts": len(forecasts),
            "scores": len(scores),
            "forecast_head_hash": forecast_head,
            "score_head_hash": previous_hash,
        }
    finally:
        connection.close()


def ledger_info(ledger: str | Path) -> dict[str, object]:
    connection = _connect(ledger)
    try:
        forecast_rows = connection.execute(
            "SELECT target_date FROM value_forecasts ORDER BY id"
        ).fetchall()
        scores = connection.execute(
            """
            SELECT error, absolute_error, covered, false_positive, false_negative
            FROM value_scores ORDER BY id
            """
        ).fetchall()
        verification = verify_ledger(ledger)
        return {
            **verification,
            "pending": len(forecast_rows) - len(scores),
            "first_target_date": str(forecast_rows[0]["target_date"]) if forecast_rows else None,
            "last_target_date": str(forecast_rows[-1]["target_date"]) if forecast_rows else None,
            "mean_bias": (
                fmean(float(row["error"]) for row in scores) if scores else None
            ),
            "mae": (
                fmean(float(row["absolute_error"]) for row in scores) if scores else None
            ),
            "coverage": (
                fmean(int(row["covered"]) for row in scores) if scores else None
            ),
            "false_positives": sum(int(row["false_positive"]) for row in scores),
            "false_negatives": sum(int(row["false_negative"]) for row in scores),
            "interpretation": (
                "Publier le forecast_head_hash avant le tirage pour ancrer la chronologie."
            ),
        }
    finally:
        connection.close()
