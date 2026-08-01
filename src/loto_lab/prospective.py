from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import UTC, date, datetime, time
from pathlib import Path
from statistics import fmean
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from . import __version__
from .domain import Draw, PrizeResult
from .probability import rank_probabilities, total_outcomes
from .value import ValueReport, _lower_rank_ev, _poisson_share_factor
from .value_backtest import _draw_jackpot, _draw_ticket_count

GENESIS_HASH = "0" * 64
PARIS_TIMEZONE = ZoneInfo("Europe/Paris")
LOTO_RECORDING_CUTOFF = time(20, 15)
EVIDENCE_FORMAT = "loto-lab.prospective-ledger"
EVIDENCE_SCHEMA_VERSION = 2
SUPPORTED_EVIDENCE_SCHEMA_VERSIONS = (1, 2)
CURRENT_LEDGER_SCHEMA_VERSION = 2

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
    score_provenance_json TEXT NOT NULL DEFAULT '{}',
    hash_version INTEGER NOT NULL DEFAULT 1 CHECK (hash_version IN (1, 2)),
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
    score_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(value_scores)").fetchall()
    }
    if "score_provenance_json" not in score_columns:
        connection.execute(
            "ALTER TABLE value_scores ADD COLUMN "
            "score_provenance_json TEXT NOT NULL DEFAULT '{}'"
        )
    if "hash_version" not in score_columns:
        connection.execute(
            "ALTER TABLE value_scores ADD COLUMN "
            "hash_version INTEGER NOT NULL DEFAULT 1 CHECK (hash_version IN (1, 2))"
        )
    connection.execute(
        """
        INSERT INTO ledger_metadata(key, value) VALUES ('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(CURRENT_LEDGER_SCHEMA_VERSION),),
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
    *,
    provenance_json: str = "{}",
    hash_version: int = 1,
) -> str:
    payload: dict[str, object] = {
        "scored_at": scored_at,
        "forecast_hash": forecast_hash,
        "previous_hash": previous_hash,
        "draw_json": draw_json,
        "metrics": metrics,
    }
    if hash_version == 2:
        payload["hash_version"] = hash_version
        payload["score_provenance_json"] = provenance_json
    elif hash_version != 1:
        raise ValueError(f"Version de hash de score inconnue: {hash_version}")
    return _digest(payload)


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
    provenance: dict[str, object],
) -> dict[str, object]:
    _validate_official_source(provenance.get("jackpot_source"))
    _validate_data_provenance(provenance.get("data"))
    if report.target_date is None:
        raise ValueError("Une prevision prospective exige une date cible")
    if report.target_date <= report.training_last_date:
        raise ValueError("La date cible doit etre posterieure aux donnees d'apprentissage")
    current_time = datetime.now(UTC)
    if not _recording_is_open(report.target_date, current_time):
        raise ValueError("Une prevision prospective ne peut pas etre retroactive")
    created_at = current_time.astimezone(UTC).isoformat()
    report_json = _canonical_json({"report": report.to_dict(), "provenance": provenance})
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


def build_data_provenance(
    paths: list[str | Path], draws: list[Draw]
) -> dict[str, object]:
    files = []
    for value in paths:
        path = Path(value)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    dated = [draw.draw_date for draw in draws if draw.draw_date is not None]
    return {
        "files": files,
        "draws_snapshot": {
            "count": len(draws),
            "first_date": min(dated).isoformat() if dated else None,
            "last_date": max(dated).isoformat() if dated else None,
            "sha256": _digest([_draw_payload(draw) for draw in draws]),
        },
    }


def _validate_official_source(value: object) -> str:
    source = str(value)
    parsed = urlparse(source)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        hostname == "fdj.fr" or hostname.endswith(".fdj.fr")
    ):
        raise ValueError("La source du resultat doit etre une URL HTTPS officielle FDJ")
    return source


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_data_provenance(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("La provenance doit contenir les empreintes des donnees")
    files = value.get("files")
    snapshot = value.get("draws_snapshot")
    if not isinstance(files, list) or not files or not isinstance(snapshot, dict):
        raise ValueError("La provenance des donnees est incomplete")
    for item in files:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("size"), int)
            or int(item["size"]) < 0
            or not _valid_sha256(item.get("sha256"))
        ):
            raise ValueError("Une empreinte de fichier est invalide")
    if (
        not isinstance(snapshot.get("count"), int)
        or int(snapshot["count"]) < 0
        or not _valid_sha256(snapshot.get("sha256"))
    ):
        raise ValueError("L'empreinte logique des tirages est invalide")
    return value


def _uses_hashed_provenance(model_version: object) -> bool:
    try:
        parts = tuple(int(part) for part in str(model_version).split(".")[:2])
    except ValueError:
        return False
    return parts >= (0, 9)


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
    *,
    provenance: dict[str, object],
    current_time: datetime | None = None,
) -> dict[str, object]:
    _validate_official_source(provenance.get("result_source"))
    _validate_data_provenance(provenance.get("data"))
    score_provenance_json = _canonical_json(provenance)
    hash_version = 2
    by_target = {
        (draw.game, draw.draw_date.isoformat()): draw
        for draw in draws
        if draw.draw_date is not None
    }
    scoring_time = current_time or datetime.now(UTC)
    if scoring_time.tzinfo is None:
        raise ValueError("L'heure de scoring doit inclure un fuseau horaire")
    scored_at = scoring_time.astimezone(UTC).isoformat()
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
            if _recording_is_open(date.fromisoformat(key[1]), scoring_time):
                skipped.append(
                    {"forecast_id": int(forecast["id"]), "reason": "tirage_non_cloture"}
                )
                continue
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
                provenance_json=score_provenance_json,
                hash_version=hash_version,
            )
            cursor = connection.execute(
                """
                INSERT INTO value_scores(
                    forecast_id, scored_at, observed_schedule_ev, error, absolute_error,
                    covered, observed_positive, false_positive, false_negative, draw_json,
                    score_provenance_json, hash_version, previous_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    score_provenance_json,
                    hash_version,
                    previous_hash,
                    record_hash,
                ),
            )
            scored.append(
                {
                    "score_id": int(cursor.lastrowid),
                    "forecast_id": int(forecast["id"]),
                    **metrics,
                    "provenance": provenance,
                    "hash_version": hash_version,
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
                provenance_json=str(row["score_provenance_json"]),
                hash_version=int(row["hash_version"]),
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
                "Publier forecast_head_hash avant le tirage, puis score_head_hash apres scoring."
            ),
        }
    finally:
        connection.close()


def export_ledger_evidence(ledger: str | Path) -> dict[str, object]:
    verification = verify_ledger(ledger)
    if not verification["valid"]:
        raise ValueError("Le registre est invalide; export refuse")
    connection = _connect(ledger)
    try:
        forecasts = []
        for row in connection.execute("SELECT * FROM value_forecasts ORDER BY id"):
            forecasts.append(
                {
                    "id": int(row["id"]),
                    "created_at": str(row["created_at"]),
                    "model_version": str(row["model_version"]),
                    "game": str(row["game"]),
                    "target_date": str(row["target_date"]),
                    "training_last_date": str(row["training_last_date"]),
                    "jackpot": float(row["jackpot"]),
                    "estimated_ev": float(row["estimated_ev"]),
                    "ev_ci_low": float(row["ev_ci_low"]),
                    "ev_ci_high": float(row["ev_ci_high"]),
                    "ticket_price": float(row["ticket_price"]),
                    "decision": str(row["decision"]),
                    "payload": json.loads(str(row["report_json"])),
                    "previous_hash": str(row["previous_hash"]),
                    "record_hash": str(row["record_hash"]),
                }
            )
        scores = []
        for row in connection.execute("SELECT * FROM value_scores ORDER BY id"):
            scores.append(
                {
                    "id": int(row["id"]),
                    "forecast_id": int(row["forecast_id"]),
                    "scored_at": str(row["scored_at"]),
                    "observed_schedule_ev": float(row["observed_schedule_ev"]),
                    "error": float(row["error"]),
                    "absolute_error": float(row["absolute_error"]),
                    "covered": int(row["covered"]),
                    "observed_positive": int(row["observed_positive"]),
                    "false_positive": int(row["false_positive"]),
                    "false_negative": int(row["false_negative"]),
                    "draw": json.loads(str(row["draw_json"])),
                    "provenance": json.loads(str(row["score_provenance_json"])),
                    "hash_version": int(row["hash_version"]),
                    "previous_hash": str(row["previous_hash"]),
                    "record_hash": str(row["record_hash"]),
                }
            )
        return {
            "format": EVIDENCE_FORMAT,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "exported_at": datetime.now(UTC).isoformat(),
            "forecasts": forecasts,
            "scores": scores,
            "ledger": ledger_info(ledger),
        }
    finally:
        connection.close()


def _draw_from_evidence(payload: dict[str, object]) -> Draw:
    draw_date = payload.get("draw_date")
    return Draw(
        tuple(int(number) for number in payload["main"]),  # type: ignore[arg-type]
        int(payload["chance"]),
        date.fromisoformat(str(draw_date)) if draw_date else None,
        str(payload["game"]),
        tuple(
            PrizeResult(
                int(prize["rank"]),
                int(prize["winners"]) if prize.get("winners") is not None else None,
                float(prize["payout"]) if prize.get("payout") is not None else None,
            )
            for prize in payload["prizes"]  # type: ignore[union-attr]
        ),
        int(payload["code_winners"]) if payload.get("code_winners") is not None else None,
        float(payload["code_payout"]) if payload.get("code_payout") is not None else None,
    )


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def verify_evidence(evidence: dict[str, object]) -> dict[str, object]:
    errors: list[str] = []
    if evidence.get("format") != EVIDENCE_FORMAT:
        errors.append("format")
    if evidence.get("schema_version") not in SUPPORTED_EVIDENCE_SCHEMA_VERSIONS:
        errors.append("schema_version")
    forecasts_value = evidence.get("forecasts")
    scores_value = evidence.get("scores")
    if not isinstance(forecasts_value, list):
        errors.append("forecasts")
        forecasts_value = []
    if not isinstance(scores_value, list):
        errors.append("scores")
        scores_value = []

    previous_hash = GENESIS_HASH
    forecast_hashes: dict[int, str] = {}
    forecast_records: dict[int, dict[str, object]] = {}
    targets: set[tuple[str, str]] = set()
    for index, item in enumerate(forecasts_value, start=1):
        label = f"forecast:{index}"
        if not isinstance(item, dict):
            errors.append(f"{label}:malformed")
            continue
        try:
            forecast_id = int(item["id"])
            payload = item["payload"]
            if not isinstance(payload, dict):
                raise TypeError
            record_hash = str(item["record_hash"])
            expected = _forecast_hash(
                created_at=str(item["created_at"]),
                model_version=str(item["model_version"]),
                game=str(item["game"]),
                target_date=str(item["target_date"]),
                training_last_date=str(item["training_last_date"]),
                jackpot=float(item["jackpot"]),
                estimated_ev=float(item["estimated_ev"]),
                ev_ci_low=float(item["ev_ci_low"]),
                ev_ci_high=float(item["ev_ci_high"]),
                ticket_price=float(item["ticket_price"]),
                decision=str(item["decision"]),
                report_json=_canonical_json(payload),
                previous_hash=previous_hash,
            )
            if str(item["previous_hash"]) != previous_hash or record_hash != expected:
                errors.append(f"{label}:hash")
            if forecast_id in forecast_hashes:
                errors.append(f"{label}:duplicate_id")
            target = (str(item["game"]), str(item["target_date"]))
            if target in targets:
                errors.append(f"{label}:duplicate_target")
            created_at = datetime.fromisoformat(str(item["created_at"]))
            target_date = date.fromisoformat(target[1])
            training_last_date = date.fromisoformat(str(item["training_last_date"]))
            if (
                created_at.tzinfo is None
                or not _recording_is_open(target_date, created_at)
                or training_last_date >= target_date
            ):
                errors.append(f"{label}:chronology")
            report = payload.get("report")
            if not isinstance(report, dict):
                errors.append(f"{label}:report")
            else:
                comparisons = {
                    "game": str(item["game"]),
                    "target_date": str(item["target_date"]),
                    "training_last_date": str(item["training_last_date"]),
                    "decision": str(item["decision"]),
                }
                if any(str(report.get(key)) != value for key, value in comparisons.items()):
                    errors.append(f"{label}:report_mismatch")
                numeric_comparisons = {
                    "jackpot": float(item["jackpot"]),
                    "estimated_ev": float(item["estimated_ev"]),
                    "ev_ci_low": float(item["ev_ci_low"]),
                    "ev_ci_high": float(item["ev_ci_high"]),
                    "ticket_price": float(item["ticket_price"]),
                }
                if any(
                    not _same_number(float(report.get(key)), value)
                    for key, value in numeric_comparisons.items()
                ):
                    errors.append(f"{label}:report_mismatch")
            if _uses_hashed_provenance(item["model_version"]):
                provenance = payload.get("provenance")
                if not isinstance(provenance, dict):
                    errors.append(f"{label}:provenance")
                else:
                    try:
                        _validate_official_source(provenance.get("jackpot_source"))
                    except ValueError:
                        errors.append(f"{label}:jackpot_source")
                    try:
                        _validate_data_provenance(provenance.get("data"))
                    except ValueError:
                        errors.append(f"{label}:data_provenance")
            forecast_hashes[forecast_id] = record_hash
            forecast_records[forecast_id] = item
            targets.add(target)
            previous_hash = record_hash
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}:malformed")
    forecast_head = previous_hash

    previous_hash = GENESIS_HASH
    scored_forecasts: set[int] = set()
    score_ids: set[int] = set()
    for index, item in enumerate(scores_value, start=1):
        label = f"score:{index}"
        if not isinstance(item, dict):
            errors.append(f"{label}:malformed")
            continue
        try:
            score_id = int(item["id"])
            forecast_id = int(item["forecast_id"])
            forecast = forecast_records[forecast_id]
            metrics: dict[str, float | int] = {
                "observed_schedule_ev": float(item["observed_schedule_ev"]),
                "error": float(item["error"]),
                "absolute_error": float(item["absolute_error"]),
                "covered": int(item["covered"]),
                "observed_positive": int(item["observed_positive"]),
                "false_positive": int(item["false_positive"]),
                "false_negative": int(item["false_negative"]),
            }
            draw_payload = item["draw"]
            if not isinstance(draw_payload, dict):
                raise TypeError
            hash_version = int(item.get("hash_version", 1))
            provenance = item.get("provenance", {})
            if not isinstance(provenance, dict):
                raise TypeError
            provenance_json = _canonical_json(provenance)
            record_hash = str(item["record_hash"])
            expected = _score_hash(
                str(item["scored_at"]),
                forecast_hashes[forecast_id],
                previous_hash,
                _canonical_json(draw_payload),
                metrics,
                provenance_json=provenance_json,
                hash_version=hash_version,
            )
            if str(item["previous_hash"]) != previous_hash or record_hash != expected:
                errors.append(f"{label}:hash")
            if forecast_id in scored_forecasts:
                errors.append(f"{label}:duplicate_forecast")
            if score_id in score_ids:
                errors.append(f"{label}:duplicate_id")
            if hash_version == 2:
                try:
                    _validate_official_source(provenance.get("result_source"))
                except ValueError:
                    errors.append(f"{label}:result_source")
                try:
                    _validate_data_provenance(provenance.get("data"))
                except ValueError:
                    errors.append(f"{label}:data_provenance")
            draw = _draw_from_evidence(draw_payload)
            scored_at = datetime.fromisoformat(str(item["scored_at"]))
            target_date = date.fromisoformat(str(forecast["target_date"]))
            if scored_at.tzinfo is None or _recording_is_open(target_date, scored_at):
                errors.append(f"{label}:chronology")
            if draw.game != forecast["game"] or draw.draw_date != target_date:
                errors.append(f"{label}:draw_target")
            observed_ev = _observed_schedule_ev(draw)
            if observed_ev is None:
                errors.append(f"{label}:incomplete_draw")
            else:
                estimated_ev = float(forecast["estimated_ev"])
                ticket_price = float(forecast["ticket_price"])
                eligible = str(forecast["decision"]) == "eligible"
                observed_positive = observed_ev > ticket_price
                expected_metrics: dict[str, float | int] = {
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
                for key, expected_value in expected_metrics.items():
                    actual = metrics[key]
                    matches = (
                        _same_number(float(actual), float(expected_value))
                        if isinstance(expected_value, float)
                        else actual == expected_value
                    )
                    if not matches:
                        errors.append(f"{label}:metric:{key}")
            scored_forecasts.add(forecast_id)
            score_ids.add(score_id)
            previous_hash = record_hash
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}:malformed")
    score_head = previous_hash

    manifest = evidence.get("ledger")
    if not isinstance(manifest, dict):
        errors.append("ledger")
    else:
        expected_manifest = {
            "forecasts": len(forecasts_value),
            "scores": len(scores_value),
            "forecast_head_hash": forecast_head,
            "score_head_hash": score_head,
        }
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            errors.append("ledger:mismatch")
    return {
        "valid": not errors,
        "errors": errors,
        "forecasts": len(forecasts_value),
        "scores": len(scores_value),
        "forecast_head_hash": forecast_head,
        "score_head_hash": score_head,
    }
