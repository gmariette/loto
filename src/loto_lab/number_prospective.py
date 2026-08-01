from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from . import __version__
from .domain import DEFAULT_RULES, Draw
from .model_identity import validate_number_model_specification
from .prospective import (
    GENESIS_HASH,
    _recording_is_open,
    _validate_data_provenance,
    _validate_official_source,
)

NUMBER_EVIDENCE_FORMAT = "loto-lab.number-prospective-ledger"
NUMBER_EVIDENCE_SCHEMA_VERSION = 2
SUPPORTED_NUMBER_EVIDENCE_SCHEMA_VERSIONS = (1, 2)
NUMBER_QUALIFICATION_SCORES = 100
NUMBER_FAMILY_ALPHA = 0.05
NUMBER_BOOTSTRAP_SIMULATIONS = 2_000

NUMBER_LEDGER_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS number_forecasts (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    evaluation_cohort TEXT NOT NULL,
    cohort_index INTEGER NOT NULL CHECK (cohort_index >= 1),
    game TEXT NOT NULL,
    target_date TEXT NOT NULL,
    training_last_date TEXT NOT NULL,
    historical_status TEXT NOT NULL,
    model TEXT NOT NULL,
    main_json TEXT NOT NULL,
    chance INTEGER NOT NULL CHECK (chance BETWEEN 1 AND 10),
    payload_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL UNIQUE,
    UNIQUE (evaluation_cohort, game, target_date),
    CHECK (target_date > training_last_date)
);

CREATE TABLE IF NOT EXISTS number_scores (
    id INTEGER PRIMARY KEY,
    forecast_id INTEGER NOT NULL UNIQUE REFERENCES number_forecasts(id),
    scored_at TEXT NOT NULL,
    main_hits INTEGER NOT NULL CHECK (main_hits BETWEEN 0 AND 5),
    chance_hit INTEGER NOT NULL CHECK (chance_hit IN (0, 1)),
    draw_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL UNIQUE
);

CREATE TRIGGER IF NOT EXISTS number_forecasts_no_update
BEFORE UPDATE ON number_forecasts BEGIN
    SELECT RAISE(ABORT, 'number_forecasts is append-only');
END;

CREATE TRIGGER IF NOT EXISTS number_forecasts_no_delete
BEFORE DELETE ON number_forecasts BEGIN
    SELECT RAISE(ABORT, 'number_forecasts is append-only');
END;

CREATE TRIGGER IF NOT EXISTS number_scores_no_update
BEFORE UPDATE ON number_scores BEGIN
    SELECT RAISE(ABORT, 'number_scores is append-only');
END;

CREATE TRIGGER IF NOT EXISTS number_scores_no_delete
BEFORE DELETE ON number_scores BEGIN
    SELECT RAISE(ABORT, 'number_scores is append-only');
END;
"""


def _normalize_json_keys(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _normalize_json_keys(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_keys(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _normalize_json_keys(value),
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
    connection.executescript(NUMBER_LEDGER_SCHEMA)
    return connection


def _as_iso_date(value: object, field: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as error:
            raise ValueError(f"Date {field} invalide") from error
    raise ValueError(f"Date {field} manquante")


def _validate_numbers_prediction(prediction: dict[str, object]) -> tuple[list[int], int]:
    if prediction.get("status") not in ("qualified", "forced_experimental"):
        raise ValueError("Seule une grille explicite peut etre enregistree")
    raw_main = prediction.get("numbers")
    if not isinstance(raw_main, (list, tuple)):
        raise ValueError("Numeros principaux manquants")
    main = [int(number) for number in raw_main]
    if len(main) != DEFAULT_RULES.main_drawn or len(set(main)) != len(main):
        raise ValueError("La grille doit contenir cinq numeros distincts")
    if any(number < 1 or number > DEFAULT_RULES.main_pool for number in main):
        raise ValueError("Numero principal hors limites")
    chance = int(prediction.get("chance", 0))
    if chance < 1 or chance > DEFAULT_RULES.chance_pool:
        raise ValueError("Numero Chance hors limites")
    return sorted(main), chance


def _forecast_hash(row: dict[str, object]) -> str:
    return _digest(
        {
            key: row[key]
            for key in (
                "created_at",
                "model_version",
                "evaluation_cohort",
                "cohort_index",
                "game",
                "target_date",
                "training_last_date",
                "historical_status",
                "model",
                "main_json",
                "chance",
                "payload_json",
                "previous_hash",
            )
        }
    )


def _score_hash(row: dict[str, object], forecast_hash: str) -> str:
    return _digest(
        {
            "forecast_hash": forecast_hash,
            **{
                key: row[key]
                for key in (
                    "scored_at",
                    "main_hits",
                    "chance_hit",
                    "draw_json",
                    "provenance_json",
                    "previous_hash",
                )
            },
        }
    )


def _cohort_index(connection: sqlite3.Connection, cohort: str) -> int:
    existing = connection.execute(
        "SELECT cohort_index FROM number_forecasts WHERE evaluation_cohort = ? LIMIT 1",
        (cohort,),
    ).fetchone()
    if existing:
        return int(existing["cohort_index"])
    row = connection.execute(
        "SELECT COALESCE(MAX(cohort_index), 0) + 1 AS next_index FROM number_forecasts"
    ).fetchone()
    return int(row["next_index"])


def cohort_alpha_budget(cohort_index: int) -> float:
    if cohort_index < 1:
        raise ValueError("L'index de cohorte doit etre positif")
    return NUMBER_FAMILY_ALPHA / (cohort_index * (cohort_index + 1))


def record_number_forecast(
    ledger: str | Path,
    prediction: dict[str, object],
    *,
    model_specification: dict[str, object],
    provenance: dict[str, object],
    model_version: str = __version__,
    current_time: datetime | None = None,
) -> dict[str, object]:
    cohort = validate_number_model_specification(model_specification)
    _validate_data_provenance(provenance.get("data"))
    main, chance = _validate_numbers_prediction(prediction)
    game = str(prediction.get("game", ""))
    model = str(prediction.get("model", ""))
    if not game or not model:
        raise ValueError("Jeu ou modele manquant")
    parameters = model_specification.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("game") != game:
        raise ValueError("La specification ne correspond pas au jeu predit")
    models = parameters.get("models")
    if not isinstance(models, list) or model not in models:
        raise ValueError("Le modele predit n'appartient pas a la specification")
    target_date = _as_iso_date(prediction.get("target_date"), "cible")
    training_last_date = _as_iso_date(
        prediction.get("training_last_date"), "d'apprentissage"
    )
    if target_date <= training_last_date:
        raise ValueError("La cible doit etre posterieure aux donnees d'apprentissage")
    now = current_time or datetime.now(UTC)
    if not _recording_is_open(date.fromisoformat(target_date), now):
        raise ValueError("Une prevision de numeros ne peut pas etre retroactive")
    created_at = now.astimezone(UTC).isoformat()
    payload_json = _canonical_json(
        {
            "prediction": prediction,
            "model_specification": model_specification,
            "provenance": provenance,
        }
    )
    connection = _connect(ledger)
    try:
        connection.execute("BEGIN IMMEDIATE")
        index = _cohort_index(connection, cohort)
        previous = connection.execute(
            "SELECT record_hash FROM number_forecasts ORDER BY id DESC LIMIT 1"
        ).fetchone()
        row: dict[str, object] = {
            "created_at": created_at,
            "model_version": model_version,
            "evaluation_cohort": cohort,
            "cohort_index": index,
            "game": game,
            "target_date": target_date,
            "training_last_date": training_last_date,
            "historical_status": str(prediction["status"]),
            "model": model,
            "main_json": _canonical_json(main),
            "chance": chance,
            "payload_json": payload_json,
            "previous_hash": str(previous["record_hash"]) if previous else GENESIS_HASH,
        }
        record_hash = _forecast_hash(row)
        try:
            cursor = connection.execute(
                """
                INSERT INTO number_forecasts(
                    created_at, model_version, evaluation_cohort, cohort_index, game,
                    target_date, training_last_date, historical_status, model, main_json,
                    chance, payload_json, previous_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*row.values(), record_hash),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"Une prevision {game} existe deja pour {target_date} dans cette cohorte"
            ) from error
        connection.commit()
        return {
            "forecast_id": int(cursor.lastrowid),
            "created_at": created_at,
            "model_version": model_version,
            "evaluation_cohort": cohort,
            "cohort_index": index,
            "alpha_budget": cohort_alpha_budget(index),
            "game": game,
            "target_date": target_date,
            "main": main,
            "chance": chance,
            "previous_hash": row["previous_hash"],
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
    }


def score_pending_number_forecasts(
    ledger: str | Path,
    draws: list[Draw],
    *,
    provenance: dict[str, object],
    current_time: datetime | None = None,
) -> dict[str, object]:
    _validate_official_source(provenance.get("result_source"))
    _validate_data_provenance(provenance.get("data"))
    now = current_time or datetime.now(UTC)
    by_target = {
        (draw.game, draw.draw_date.isoformat()): draw
        for draw in draws
        if draw.draw_date is not None
    }
    connection = _connect(ledger)
    new_scores: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    try:
        connection.execute("BEGIN IMMEDIATE")
        pending = connection.execute(
            """
            SELECT f.* FROM number_forecasts f
            LEFT JOIN number_scores s ON s.forecast_id = f.id
            WHERE s.id IS NULL ORDER BY f.id
            """
        ).fetchall()
        previous = connection.execute(
            "SELECT record_hash FROM number_scores ORDER BY id DESC LIMIT 1"
        ).fetchone()
        previous_hash = str(previous["record_hash"]) if previous else GENESIS_HASH
        provenance_json = _canonical_json(provenance)
        for forecast in pending:
            target = (str(forecast["game"]), str(forecast["target_date"]))
            draw = by_target.get(target)
            if draw is None:
                skipped.append({"forecast_id": int(forecast["id"]), "reason": "absent"})
                continue
            if _recording_is_open(draw.draw_date, now):
                skipped.append(
                    {"forecast_id": int(forecast["id"]), "reason": "tirage_non_cloture"}
                )
                continue
            selected = set(json.loads(str(forecast["main_json"])))
            row: dict[str, object] = {
                "scored_at": now.astimezone(UTC).isoformat(),
                "main_hits": len(selected.intersection(draw.main)),
                "chance_hit": int(int(forecast["chance"]) == draw.chance),
                "draw_json": _canonical_json(_draw_payload(draw)),
                "provenance_json": provenance_json,
                "previous_hash": previous_hash,
            }
            record_hash = _score_hash(row, str(forecast["record_hash"]))
            cursor = connection.execute(
                """
                INSERT INTO number_scores(
                    forecast_id, scored_at, main_hits, chance_hit, draw_json,
                    provenance_json, previous_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(forecast["id"]), *row.values(), record_hash),
            )
            previous_hash = record_hash
            new_scores.append(
                {
                    "score_id": int(cursor.lastrowid),
                    "forecast_id": int(forecast["id"]),
                    "main_hits": row["main_hits"],
                    "chance_hit": bool(row["chance_hit"]),
                    "record_hash": record_hash,
                }
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {"new_scores": len(new_scores), "scores": new_scores, "skipped": skipped}


def _exact_top5_tail(draw_count: int, observed_total: int) -> float:
    if draw_count < 1:
        raise ValueError("Le nombre de tirages doit etre positif")
    denominator = math.comb(DEFAULT_RULES.main_pool, DEFAULT_RULES.main_drawn)
    single = np.asarray(
        [
            math.comb(DEFAULT_RULES.main_drawn, hits)
            * math.comb(
                DEFAULT_RULES.main_pool - DEFAULT_RULES.main_drawn,
                DEFAULT_RULES.main_drawn - hits,
            )
            / denominator
            for hits in range(DEFAULT_RULES.main_drawn + 1)
        ]
    )
    distribution = np.asarray([1.0])
    for _ in range(draw_count):
        distribution = np.convolve(distribution, single)
    return float(distribution[observed_total:].sum())


def _uplift_interval(hits: np.ndarray, seed: int) -> tuple[float, float]:
    expected = DEFAULT_RULES.main_drawn**2 / DEFAULT_RULES.main_pool
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [
            rng.choice(hits, size=len(hits), replace=True).mean() - expected
            for _ in range(NUMBER_BOOTSTRAP_SIMULATIONS)
        ]
    )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _cohort_summaries(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT f.id, f.evaluation_cohort, f.cohort_index, f.model_version, f.model,
               f.target_date, s.main_hits, s.chance_hit
        FROM number_forecasts f
        LEFT JOIN number_scores s ON s.forecast_id = f.id
        ORDER BY f.cohort_index, f.target_date, f.id
        """
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["evaluation_cohort"])].append(row)
    summaries = []
    expected = DEFAULT_RULES.main_drawn**2 / DEFAULT_RULES.main_pool
    for cohort, cohort_rows in grouped.items():
        index = int(cohort_rows[0]["cohort_index"])
        scored = [row for row in cohort_rows if row["main_hits"] is not None]
        qualification_rows = scored[:NUMBER_QUALIFICATION_SCORES]
        hits = np.asarray([int(row["main_hits"]) for row in qualification_rows])
        alpha = cohort_alpha_budget(index)
        if len(hits) < NUMBER_QUALIFICATION_SCORES:
            status = "insufficient_data"
            mean_hits = float(hits.mean()) if len(hits) else None
            uplift = mean_hits - expected if mean_hits is not None else None
            ci_low = ci_high = p_value = None
        else:
            mean_hits = float(hits.mean())
            uplift = mean_hits - expected
            ci_low, ci_high = _uplift_interval(hits, 20_260_801 + index)
            p_value = _exact_top5_tail(len(hits), int(hits.sum()))
            status = "qualified" if ci_low > 0 and p_value < alpha else "not_qualified"
        summaries.append(
            {
                "evaluation_cohort": cohort,
                "cohort_index": index,
                "model_versions": sorted({str(row["model_version"]) for row in cohort_rows}),
                "models": sorted({str(row["model"]) for row in cohort_rows}),
                "forecasts": len(cohort_rows),
                "scores": len(scored),
                "qualification_scores": len(qualification_rows),
                "qualification_target": NUMBER_QUALIFICATION_SCORES,
                "alpha_budget": alpha,
                "mean_main_hits": mean_hits,
                "uniform_expected_main_hits": expected,
                "main_hit_uplift": uplift,
                "uplift_ci_low": ci_low,
                "uplift_ci_high": ci_high,
                "exact_p_value": p_value,
                "chance_hits": sum(int(row["chance_hit"]) for row in scored),
                "qualification_status": status,
            }
        )
    return summaries


def verify_number_ledger(ledger: str | Path) -> dict[str, object]:
    connection = _connect(ledger)
    errors: list[str] = []
    try:
        forecasts = connection.execute("SELECT * FROM number_forecasts ORDER BY id").fetchall()
        expected_previous = GENESIS_HASH
        cohort_indices: dict[str, int] = {}
        forecast_hashes: dict[int, str] = {}
        for row in forecasts:
            values = dict(row)
            if str(row["previous_hash"]) != expected_previous:
                errors.append(f"forecast:{row['id']}:previous_hash")
            expected_hash = _forecast_hash(values)
            if str(row["record_hash"]) != expected_hash:
                errors.append(f"forecast:{row['id']}:record_hash")
            cohort = str(row["evaluation_cohort"])
            index = int(row["cohort_index"])
            if cohort in cohort_indices and cohort_indices[cohort] != index:
                errors.append(f"forecast:{row['id']}:cohort_index")
            cohort_indices[cohort] = index
            forecast_hashes[int(row["id"])] = str(row["record_hash"])
            expected_previous = str(row["record_hash"])
        scores = connection.execute("SELECT * FROM number_scores ORDER BY id").fetchall()
        expected_score_previous = GENESIS_HASH
        for row in scores:
            values = dict(row)
            if str(row["previous_hash"]) != expected_score_previous:
                errors.append(f"score:{row['id']}:previous_hash")
            forecast_hash = forecast_hashes.get(int(row["forecast_id"]))
            if forecast_hash is None or str(row["record_hash"]) != _score_hash(
                values, forecast_hash or ""
            ):
                errors.append(f"score:{row['id']}:record_hash")
            expected_score_previous = str(row["record_hash"])
        if sorted(set(cohort_indices.values())) != list(range(1, len(cohort_indices) + 1)):
            errors.append("cohorts:index_sequence")
        return {
            "valid": not errors,
            "errors": errors,
            "forecasts": len(forecasts),
            "scores": len(scores),
            "forecast_head_hash": expected_previous,
            "score_head_hash": expected_score_previous,
            "cohorts": _cohort_summaries(connection),
        }
    finally:
        connection.close()


def export_number_evidence(ledger: str | Path) -> dict[str, object]:
    connection = _connect(ledger)
    try:
        forecasts = []
        for row in connection.execute("SELECT * FROM number_forecasts ORDER BY id"):
            value = dict(row)
            value["main"] = json.loads(value.pop("main_json"))
            payload_json = value.pop("payload_json")
            value["payload"] = json.loads(payload_json)
            value["payload_canonical_json"] = payload_json
            forecasts.append(value)
        scores = []
        for row in connection.execute("SELECT * FROM number_scores ORDER BY id"):
            value = dict(row)
            value["draw"] = json.loads(value.pop("draw_json"))
            value["provenance"] = json.loads(value.pop("provenance_json"))
            scores.append(value)
    finally:
        connection.close()
    return {
        "format": NUMBER_EVIDENCE_FORMAT,
        "schema_version": NUMBER_EVIDENCE_SCHEMA_VERSION,
        "ledger": verify_number_ledger(ledger),
        "forecasts": forecasts,
        "scores": scores,
    }


def verify_number_evidence(evidence: object) -> dict[str, object]:
    errors: list[str] = []
    if not isinstance(evidence, dict):
        return {"valid": False, "errors": ["evidence:not_object"]}
    if evidence.get("format") != NUMBER_EVIDENCE_FORMAT:
        errors.append("evidence:format")
    schema_version = evidence.get("schema_version")
    if schema_version not in SUPPORTED_NUMBER_EVIDENCE_SCHEMA_VERSIONS:
        errors.append("evidence:schema_version")
    forecasts = evidence.get("forecasts")
    scores = evidence.get("scores")
    claimed = evidence.get("ledger")
    if not isinstance(forecasts, list) or not isinstance(scores, list):
        return {"valid": False, "errors": [*errors, "evidence:records"]}
    if not isinstance(claimed, dict):
        return {"valid": False, "errors": [*errors, "evidence:ledger"]}

    forecast_hashes: dict[int, str] = {}
    normalized_forecasts: list[dict[str, object]] = []
    previous_hash = GENESIS_HASH
    for position, forecast in enumerate(forecasts, start=1):
        if not isinstance(forecast, dict):
            errors.append(f"forecast:{position}:not_object")
            continue
        try:
            row = dict(forecast)
            row["main_json"] = _canonical_json(row.pop("main"))
            payload = row.pop("payload")
            payload_json = row.pop("payload_canonical_json", None)
            if payload_json is None:
                row["payload_json"] = _canonical_json(payload)
            elif not isinstance(payload_json, str):
                raise ValueError("Payload canonique invalide")
            else:
                if _canonical_json(json.loads(payload_json)) != _canonical_json(payload):
                    errors.append(f"forecast:{position}:payload_mismatch")
                row["payload_json"] = payload_json
            if row["previous_hash"] != previous_hash:
                errors.append(f"forecast:{position}:previous_hash")
            expected_hash = _forecast_hash(row)
            if row["record_hash"] != expected_hash:
                errors.append(f"forecast:{position}:record_hash")
            previous_hash = str(row["record_hash"])
            forecast_hashes[int(row["id"])] = previous_hash
            normalized_forecasts.append(row)
        except (KeyError, TypeError, ValueError):
            errors.append(f"forecast:{position}:invalid")

    score_previous_hash = GENESIS_HASH
    normalized_scores: list[dict[str, object]] = []
    for position, score in enumerate(scores, start=1):
        if not isinstance(score, dict):
            errors.append(f"score:{position}:not_object")
            continue
        try:
            row = dict(score)
            row["draw_json"] = _canonical_json(row.pop("draw"))
            row["provenance_json"] = _canonical_json(row.pop("provenance"))
            if row["previous_hash"] != score_previous_hash:
                errors.append(f"score:{position}:previous_hash")
            forecast_hash = forecast_hashes[int(row["forecast_id"])]
            expected_hash = _score_hash(row, forecast_hash)
            if row["record_hash"] != expected_hash:
                errors.append(f"score:{position}:record_hash")
            score_previous_hash = str(row["record_hash"])
            normalized_scores.append(row)
        except (KeyError, TypeError, ValueError):
            errors.append(f"score:{position}:invalid")

    if claimed.get("forecasts") != len(forecasts):
        errors.append("ledger:forecast_count")
    if claimed.get("scores") != len(scores):
        errors.append("ledger:score_count")
    if claimed.get("forecast_head_hash") != previous_hash:
        errors.append("ledger:forecast_head_hash")
    if claimed.get("score_head_hash") != score_previous_hash:
        errors.append("ledger:score_head_hash")
    if len(normalized_forecasts) == len(forecasts) and len(normalized_scores) == len(scores):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        try:
            connection.executescript(NUMBER_LEDGER_SCHEMA)
            for row in normalized_forecasts:
                connection.execute(
                    """
                    INSERT INTO number_forecasts(
                        id, created_at, model_version, evaluation_cohort, cohort_index,
                        game, target_date, training_last_date, historical_status, model,
                        main_json, chance, payload_json, previous_hash, record_hash
                    ) VALUES (
                        :id, :created_at, :model_version, :evaluation_cohort, :cohort_index,
                        :game, :target_date, :training_last_date, :historical_status, :model,
                        :main_json, :chance, :payload_json, :previous_hash, :record_hash
                    )
                    """,
                    row,
                )
            for row in normalized_scores:
                connection.execute(
                    """
                    INSERT INTO number_scores(
                        id, forecast_id, scored_at, main_hits, chance_hit, draw_json,
                        provenance_json, previous_hash, record_hash
                    ) VALUES (
                        :id, :forecast_id, :scored_at, :main_hits, :chance_hit, :draw_json,
                        :provenance_json, :previous_hash, :record_hash
                    )
                    """,
                    row,
                )
            if _canonical_json(claimed.get("cohorts")) != _canonical_json(
                _cohort_summaries(connection)
            ):
                errors.append("ledger:cohorts")
        except sqlite3.Error:
            errors.append("evidence:database_constraints")
        finally:
            connection.close()
    return {
        "valid": not errors,
        "errors": errors,
        "forecasts": len(forecasts),
        "scores": len(scores),
        "forecast_head_hash": previous_hash,
        "score_head_hash": score_previous_hash,
    }
