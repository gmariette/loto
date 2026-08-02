from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, date, datetime
from math import ceil
from pathlib import Path

import numpy as np

from . import __version__
from .domain import Draw, PrizeResult
from .model_identity import validate_popularity_model_specification
from .popularity import (
    FEATURE_NAMES,
    PopularityPredictor,
    _feature_matrix,
    _poisson_deviance,
    popularity_observations,
)
from .probability import rank_probabilities, total_outcomes
from .prospective import (
    GENESIS_HASH,
    _recording_is_open,
    _validate_data_provenance,
    _validate_official_source,
)

POPULARITY_EVIDENCE_FORMAT = "loto-lab.popularity-prospective-ledger"
POPULARITY_EVIDENCE_SCHEMA_VERSION = 1
POPULARITY_QUALIFICATION_SCORES = 100
POPULARITY_FAMILY_ALPHA = 0.05
POPULARITY_INFERENCE_SIMULATIONS = 2_000
POPULARITY_INFERENCE_BLOCK_SIZE = 12
POPULARITY_INFERENCE_SEED = 20_260_803

POPULARITY_LEDGER_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS popularity_snapshots (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    evaluation_cohort TEXT NOT NULL,
    cohort_index INTEGER NOT NULL CHECK (cohort_index >= 1),
    game TEXT NOT NULL,
    target_date TEXT NOT NULL,
    training_last_date TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL UNIQUE,
    UNIQUE (evaluation_cohort, game, target_date),
    CHECK (target_date > training_last_date)
);

CREATE TABLE IF NOT EXISTS popularity_scores (
    id INTEGER PRIMARY KEY,
    snapshot_id INTEGER NOT NULL UNIQUE REFERENCES popularity_snapshots(id),
    scored_at TEXT NOT NULL,
    jackpot_winners INTEGER NOT NULL CHECK (jackpot_winners >= 0),
    estimated_tickets REAL NOT NULL CHECK (estimated_tickets > 0),
    exposure REAL NOT NULL CHECK (exposure > 0),
    predicted_winners REAL NOT NULL CHECK (predicted_winners > 0),
    baseline_predicted_winners REAL NOT NULL CHECK (baseline_predicted_winners > 0),
    model_deviance REAL NOT NULL CHECK (model_deviance >= 0),
    baseline_deviance REAL NOT NULL CHECK (baseline_deviance >= 0),
    deviance_delta REAL NOT NULL,
    draw_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL UNIQUE
);

CREATE TRIGGER IF NOT EXISTS popularity_snapshots_no_update
BEFORE UPDATE ON popularity_snapshots BEGIN
    SELECT RAISE(ABORT, 'popularity_snapshots is append-only');
END;

CREATE TRIGGER IF NOT EXISTS popularity_snapshots_no_delete
BEFORE DELETE ON popularity_snapshots BEGIN
    SELECT RAISE(ABORT, 'popularity_snapshots is append-only');
END;

CREATE TRIGGER IF NOT EXISTS popularity_scores_no_update
BEFORE UPDATE ON popularity_scores BEGIN
    SELECT RAISE(ABORT, 'popularity_scores is append-only');
END;

CREATE TRIGGER IF NOT EXISTS popularity_scores_no_delete
BEFORE DELETE ON popularity_scores BEGIN
    SELECT RAISE(ABORT, 'popularity_scores is append-only');
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
    connection.executescript(POPULARITY_LEDGER_SCHEMA)
    return connection


def _snapshot_hash(row: dict[str, object]) -> str:
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
                "payload_json",
                "previous_hash",
            )
        }
    )


def _score_hash(row: dict[str, object], snapshot_hash: str) -> str:
    return _digest(
        {
            "snapshot_hash": snapshot_hash,
            **{
                key: row[key]
                for key in (
                    "scored_at",
                    "jackpot_winners",
                    "estimated_tickets",
                    "exposure",
                    "predicted_winners",
                    "baseline_predicted_winners",
                    "model_deviance",
                    "baseline_deviance",
                    "deviance_delta",
                    "draw_json",
                    "provenance_json",
                    "previous_hash",
                )
            },
        }
    )


def _cohort_index(connection: sqlite3.Connection, cohort: str) -> int:
    existing = connection.execute(
        "SELECT cohort_index FROM popularity_snapshots WHERE evaluation_cohort = ? LIMIT 1",
        (cohort,),
    ).fetchone()
    if existing:
        return int(existing["cohort_index"])
    row = connection.execute(
        "SELECT COALESCE(MAX(cohort_index), 0) + 1 AS value FROM popularity_snapshots"
    ).fetchone()
    return int(row["value"])


def popularity_cohort_alpha_budget(cohort_index: int) -> float:
    if cohort_index < 1:
        raise ValueError("L'index de cohorte doit etre positif")
    return POPULARITY_FAMILY_ALPHA / (cohort_index * (cohort_index + 1))


def serialize_popularity_predictor(predictor: PopularityPredictor) -> dict[str, object]:
    raw_coefficients = predictor.model.coef_ / predictor.scaler.scale_
    raw_intercept = predictor.model.intercept_ - np.dot(
        predictor.model.coef_, predictor.scaler.mean_ / predictor.scaler.scale_
    )
    return {
        "feature_names": list(FEATURE_NAMES),
        "raw_intercept": float(raw_intercept),
        "raw_coefficients": [float(value) for value in raw_coefficients],
        "baseline_multiplier": predictor.baseline_multiplier,
        "alpha": predictor.alpha,
        "observations": predictor.observations,
        "bootstrap_models": len(predictor.bootstrap_intercepts),
        "uncertainty_quantile": predictor.uncertainty_quantile,
        "historical_validation": predictor.validation.to_dict(),
    }


def _validate_predictor_specification(
    predictor: PopularityPredictor, parameters: dict[str, object]
) -> None:
    validation = predictor.validation
    expected_min_train = validation.observations - validation.test_observations
    checks = (
        validation.game == parameters.get("game"),
        expected_min_train == parameters.get("min_train"),
        validation.outer_folds == parameters.get("outer_folds"),
        validation.temporal_block_size
        == min(int(parameters.get("block_size", 0)), validation.test_observations),
        len(predictor.bootstrap_intercepts) == parameters.get("bootstrap_models"),
        predictor.uncertainty_quantile == parameters.get("uncertainty_quantile"),
    )
    if not all(checks):
        raise ValueError("Le modele ne correspond pas a sa specification scientifique")


def _validate_serialized_model(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Modele de popularite serialise invalide")
    if value.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Variables du modele de popularite incompatibles")
    coefficients = value.get("raw_coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != len(FEATURE_NAMES):
        raise ValueError("Coefficients du modele de popularite invalides")
    numeric = [
        value.get("raw_intercept"),
        value.get("baseline_multiplier"),
        *coefficients,
    ]
    if any(not isinstance(item, (int, float)) or not np.isfinite(item) for item in numeric):
        raise ValueError("Parametres non finis dans le modele de popularite")
    if float(value["baseline_multiplier"]) <= 0:
        raise ValueError("Reference de popularite invalide")
    return value


def record_popularity_snapshot(
    ledger: str | Path,
    predictor: PopularityPredictor,
    draws: list[Draw],
    target_date: date,
    *,
    model_specification: dict[str, object],
    provenance: dict[str, object],
    model_version: str = __version__,
    current_time: datetime | None = None,
) -> dict[str, object]:
    cohort = validate_popularity_model_specification(model_specification)
    _validate_data_provenance(provenance.get("data"))
    parameters = model_specification.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Parametres scientifiques de popularite manquants")
    _validate_predictor_specification(predictor, parameters)
    game = str(parameters.get("game", ""))
    observations = popularity_observations(draws, game)
    if not observations or observations[-1].draw.draw_date is None:
        raise ValueError("Aucune observation de popularite datee")
    if predictor.observations != len(observations):
        raise ValueError("Le modele ne correspond pas aux donnees fournies")
    if not predictor.validation.qualified:
        raise ValueError("Le modele de popularite historique n'est pas qualifie")
    training_last_date = observations[-1].draw.draw_date
    if target_date <= training_last_date:
        raise ValueError("La cible doit etre posterieure aux donnees d'apprentissage")
    now = current_time or datetime.now(UTC)
    if not _recording_is_open(target_date, now):
        raise ValueError("Un modele de popularite ne peut pas etre fige retroactivement")
    created_at = now.astimezone(UTC).isoformat()
    serialized = serialize_popularity_predictor(predictor)
    _validate_serialized_model(serialized)
    payload_json = _canonical_json(
        {
            "model": serialized,
            "model_specification": model_specification,
            "provenance": provenance,
        }
    )
    connection = _connect(ledger)
    try:
        connection.execute("BEGIN IMMEDIATE")
        index = _cohort_index(connection, cohort)
        previous = connection.execute(
            "SELECT record_hash FROM popularity_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        row: dict[str, object] = {
            "created_at": created_at,
            "model_version": model_version,
            "evaluation_cohort": cohort,
            "cohort_index": index,
            "game": game,
            "target_date": target_date.isoformat(),
            "training_last_date": training_last_date.isoformat(),
            "payload_json": payload_json,
            "previous_hash": str(previous["record_hash"]) if previous else GENESIS_HASH,
        }
        record_hash = _snapshot_hash(row)
        try:
            cursor = connection.execute(
                """
                INSERT INTO popularity_snapshots(
                    created_at, model_version, evaluation_cohort, cohort_index, game,
                    target_date, training_last_date, payload_json, previous_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*row.values(), record_hash),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"Un modele {game} existe deja pour {target_date} dans cette cohorte"
            ) from error
        connection.commit()
        return {
            "snapshot_id": int(cursor.lastrowid),
            "created_at": created_at,
            "model_version": model_version,
            "evaluation_cohort": cohort,
            "cohort_index": index,
            "alpha_budget": popularity_cohort_alpha_budget(index),
            "game": game,
            "target_date": target_date.isoformat(),
            "training_last_date": training_last_date.isoformat(),
            "historical_validation": predictor.validation.to_dict(),
            "previous_hash": row["previous_hash"],
            "record_hash": record_hash,
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _draw_popularity_values(draw: Draw) -> tuple[int, float, float]:
    prizes = {prize.rank: prize for prize in draw.prizes}
    rank_1 = prizes.get(1)
    rank_9 = prizes.get(9)
    if rank_1 is None or rank_1.winners is None:
        raise ValueError("Nombre de gagnants du rang 1 manquant")
    if rank_9 is None or rank_9.winners is None or rank_9.winners <= 0:
        raise ValueError("Nombre de gagnants du rang 9 manquant")
    rank_9_probability = {item.rank: item.probability for item in rank_probabilities()}[9]
    estimated_tickets = rank_9.winners / rank_9_probability
    return rank_1.winners, estimated_tickets, estimated_tickets / total_outcomes()


def _prediction_for_draw(model: dict[str, object], draw: Draw) -> tuple[float, float]:
    validated = _validate_serialized_model(model)
    features = _feature_matrix(
        np.asarray([draw.main], dtype=int), np.asarray([draw.chance], dtype=int)
    )[0]
    multiplier = float(
        np.exp(
            float(validated["raw_intercept"])
            + features @ np.asarray(validated["raw_coefficients"], dtype=float)
        )
    )
    return multiplier, float(validated["baseline_multiplier"])


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
    }


def score_pending_popularity_snapshots(
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
    scores: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    try:
        connection.execute("BEGIN IMMEDIATE")
        pending = connection.execute(
            """
            SELECT f.* FROM popularity_snapshots f
            LEFT JOIN popularity_scores s ON s.snapshot_id = f.id
            WHERE s.id IS NULL ORDER BY f.id
            """
        ).fetchall()
        previous = connection.execute(
            "SELECT record_hash FROM popularity_scores ORDER BY id DESC LIMIT 1"
        ).fetchone()
        previous_hash = str(previous["record_hash"]) if previous else GENESIS_HASH
        provenance_json = _canonical_json(provenance)
        for snapshot in pending:
            target = (str(snapshot["game"]), str(snapshot["target_date"]))
            draw = by_target.get(target)
            if draw is None:
                skipped.append({"snapshot_id": int(snapshot["id"]), "reason": "absent"})
                continue
            if _recording_is_open(draw.draw_date, now):
                skipped.append(
                    {"snapshot_id": int(snapshot["id"]), "reason": "tirage_non_cloture"}
                )
                continue
            try:
                actual, estimated_tickets, exposure = _draw_popularity_values(draw)
            except ValueError:
                skipped.append(
                    {"snapshot_id": int(snapshot["id"]), "reason": "rapports_incomplets"}
                )
                continue
            payload = json.loads(str(snapshot["payload_json"]))
            multiplier, baseline_multiplier = _prediction_for_draw(payload["model"], draw)
            predicted = multiplier * exposure
            baseline = baseline_multiplier * exposure
            model_deviance = float(
                _poisson_deviance(np.asarray([actual]), np.asarray([predicted]))[0]
            )
            baseline_deviance = float(
                _poisson_deviance(np.asarray([actual]), np.asarray([baseline]))[0]
            )
            row: dict[str, object] = {
                "scored_at": now.astimezone(UTC).isoformat(),
                "jackpot_winners": actual,
                "estimated_tickets": estimated_tickets,
                "exposure": exposure,
                "predicted_winners": predicted,
                "baseline_predicted_winners": baseline,
                "model_deviance": model_deviance,
                "baseline_deviance": baseline_deviance,
                "deviance_delta": model_deviance - baseline_deviance,
                "draw_json": _canonical_json(_draw_payload(draw)),
                "provenance_json": provenance_json,
                "previous_hash": previous_hash,
            }
            record_hash = _score_hash(row, str(snapshot["record_hash"]))
            cursor = connection.execute(
                """
                INSERT INTO popularity_scores(
                    snapshot_id, scored_at, jackpot_winners, estimated_tickets, exposure,
                    predicted_winners, baseline_predicted_winners, model_deviance,
                    baseline_deviance, deviance_delta, draw_json, provenance_json,
                    previous_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(snapshot["id"]), *row.values(), record_hash),
            )
            previous_hash = record_hash
            scores.append(
                {
                    "score_id": int(cursor.lastrowid),
                    "snapshot_id": int(snapshot["id"]),
                    "jackpot_winners": actual,
                    "predicted_winners": predicted,
                    "baseline_predicted_winners": baseline,
                    "deviance_delta": row["deviance_delta"],
                    "record_hash": record_hash,
                }
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {"new_scores": len(scores), "scores": scores, "skipped": skipped}


def _prospective_inference(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    block_size = min(POPULARITY_INFERENCE_BLOCK_SIZE, len(values))
    blocks_needed = ceil(len(values) / block_size)
    rng = np.random.default_rng(seed)
    means = []
    extreme = 0
    observed = float(values.mean())
    for _ in range(POPULARITY_INFERENCE_SIMULATIONS):
        starts = rng.integers(0, len(values) - block_size + 1, size=blocks_needed)
        sample = np.concatenate(
            [values[start : start + block_size] for start in starts]
        )[: len(values)]
        means.append(float(sample.mean()))
        signs = np.repeat(
            rng.choice((-1.0, 1.0), size=blocks_needed), block_size
        )[: len(values)]
        if float((values * signs).mean()) <= observed:
            extreme += 1
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
        (extreme + 1) / (POPULARITY_INFERENCE_SIMULATIONS + 1),
    )


def _cohort_summaries(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT f.evaluation_cohort, f.cohort_index, f.model_version, f.target_date,
               s.model_deviance, s.baseline_deviance, s.deviance_delta
        FROM popularity_snapshots f
        LEFT JOIN popularity_scores s ON s.snapshot_id = f.id
        ORDER BY f.cohort_index, f.target_date, f.id
        """
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["evaluation_cohort"])].append(row)
    summaries = []
    for cohort, cohort_rows in grouped.items():
        index = int(cohort_rows[0]["cohort_index"])
        scored = [row for row in cohort_rows if row["deviance_delta"] is not None]
        qualification = scored[:POPULARITY_QUALIFICATION_SCORES]
        alpha = popularity_cohort_alpha_budget(index)
        if qualification:
            model_mean = float(np.mean([row["model_deviance"] for row in qualification]))
            baseline_mean = float(
                np.mean([row["baseline_deviance"] for row in qualification])
            )
            mean_delta = float(np.mean([row["deviance_delta"] for row in qualification]))
        else:
            model_mean = baseline_mean = mean_delta = None
        if len(qualification) < POPULARITY_QUALIFICATION_SCORES:
            low = high = p_value = None
            status = "insufficient_data"
        else:
            values = np.asarray([row["deviance_delta"] for row in qualification], dtype=float)
            low, high, p_value = _prospective_inference(
                values, POPULARITY_INFERENCE_SEED + index
            )
            status = "qualified" if high < 0 and p_value < alpha else "not_qualified"
        summaries.append(
            {
                "evaluation_cohort": cohort,
                "cohort_index": index,
                "model_versions": sorted(
                    {str(row["model_version"]) for row in cohort_rows}
                ),
                "snapshots": len(cohort_rows),
                "scores": len(scored),
                "qualification_scores": len(qualification),
                "qualification_target": POPULARITY_QUALIFICATION_SCORES,
                "alpha_budget": alpha,
                "mean_model_deviance": model_mean,
                "mean_baseline_deviance": baseline_mean,
                "mean_deviance_delta": mean_delta,
                "delta_ci_low": low,
                "delta_ci_high": high,
                "block_sign_p_value": p_value,
                "qualification_status": status,
            }
        )
    return summaries


def _verify_connection(connection: sqlite3.Connection) -> dict[str, object]:
    errors: list[str] = []
    snapshots = connection.execute("SELECT * FROM popularity_snapshots ORDER BY id").fetchall()
    previous_hash = GENESIS_HASH
    snapshot_hashes: dict[int, str] = {}
    snapshot_rows: dict[int, sqlite3.Row] = {}
    snapshot_models: dict[int, dict[str, object]] = {}
    cohort_indices: dict[str, int] = {}
    for row in snapshots:
        values = dict(row)
        if str(row["previous_hash"]) != previous_hash:
            errors.append(f"snapshot:{row['id']}:previous_hash")
        if str(row["record_hash"]) != _snapshot_hash(values):
            errors.append(f"snapshot:{row['id']}:record_hash")
        try:
            payload = json.loads(str(row["payload_json"]))
            cohort = validate_popularity_model_specification(payload["model_specification"])
            model = _validate_serialized_model(payload["model"])
            parameters = payload["model_specification"].get("parameters")
            if not isinstance(parameters, dict) or parameters.get("game") != row["game"]:
                errors.append(f"snapshot:{row['id']}:game")
            _validate_data_provenance(payload["provenance"].get("data"))
            if cohort != row["evaluation_cohort"]:
                errors.append(f"snapshot:{row['id']}:model_identity")
            snapshot_models[int(row["id"])] = model
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            errors.append(f"snapshot:{row['id']}:payload")
        cohort = str(row["evaluation_cohort"])
        index = int(row["cohort_index"])
        if cohort in cohort_indices and cohort_indices[cohort] != index:
            errors.append(f"snapshot:{row['id']}:cohort_index")
        cohort_indices[cohort] = index
        snapshot_hashes[int(row["id"])] = str(row["record_hash"])
        snapshot_rows[int(row["id"])] = row
        previous_hash = str(row["record_hash"])

    scores = connection.execute("SELECT * FROM popularity_scores ORDER BY id").fetchall()
    score_previous_hash = GENESIS_HASH
    for row in scores:
        values = dict(row)
        if str(row["previous_hash"]) != score_previous_hash:
            errors.append(f"score:{row['id']}:previous_hash")
        snapshot_hash = snapshot_hashes.get(int(row["snapshot_id"]))
        if snapshot_hash is None or str(row["record_hash"]) != _score_hash(
            values, snapshot_hash or ""
        ):
            errors.append(f"score:{row['id']}:record_hash")
        try:
            snapshot_id = int(row["snapshot_id"])
            snapshot = snapshot_rows[snapshot_id]
            model = snapshot_models[snapshot_id]
            draw_value = json.loads(str(row["draw_json"]))
            draw = Draw(
                tuple(int(value) for value in draw_value["main"]),
                int(draw_value["chance"]),
                date.fromisoformat(draw_value["draw_date"]),
                str(draw_value["game"]),
                tuple(
                    PrizeResult(
                        int(prize["rank"]),
                        None if prize["winners"] is None else int(prize["winners"]),
                        None if prize["payout"] is None else float(prize["payout"]),
                    )
                    for prize in draw_value["prizes"]
                ),
            )
            if draw.game != snapshot["game"] or draw.draw_date.isoformat() != snapshot[
                "target_date"
            ]:
                errors.append(f"score:{row['id']}:target")
            actual, estimated_tickets, exposure = _draw_popularity_values(draw)
            multiplier, baseline_multiplier = _prediction_for_draw(model, draw)
            predicted = multiplier * exposure
            baseline = baseline_multiplier * exposure
            model_deviance = float(
                _poisson_deviance(np.asarray([actual]), np.asarray([predicted]))[0]
            )
            baseline_deviance = float(
                _poisson_deviance(np.asarray([actual]), np.asarray([baseline]))[0]
            )
            expected_values = (
                (actual, row["jackpot_winners"]),
                (estimated_tickets, row["estimated_tickets"]),
                (exposure, row["exposure"]),
                (predicted, row["predicted_winners"]),
                (baseline, row["baseline_predicted_winners"]),
                (model_deviance, row["model_deviance"]),
                (baseline_deviance, row["baseline_deviance"]),
                (model_deviance - baseline_deviance, row["deviance_delta"]),
            )
            if any(
                not np.isclose(float(expected), float(observed), rtol=1e-12, atol=1e-12)
                for expected, observed in expected_values
            ):
                errors.append(f"score:{row['id']}:metrics")
            provenance = json.loads(str(row["provenance_json"]))
            _validate_official_source(provenance.get("result_source"))
            _validate_data_provenance(provenance.get("data"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            errors.append(f"score:{row['id']}:payload")
        score_previous_hash = str(row["record_hash"])
    if sorted(set(cohort_indices.values())) != list(range(1, len(cohort_indices) + 1)):
        errors.append("cohorts:index_sequence")
    return {
        "valid": not errors,
        "errors": errors,
        "snapshots": len(snapshots),
        "scores": len(scores),
        "snapshot_head_hash": previous_hash,
        "score_head_hash": score_previous_hash,
        "cohorts": _cohort_summaries(connection),
    }


def verify_popularity_ledger(ledger: str | Path) -> dict[str, object]:
    connection = _connect(ledger)
    try:
        return _verify_connection(connection)
    finally:
        connection.close()


def export_popularity_evidence(ledger: str | Path) -> dict[str, object]:
    connection = _connect(ledger)
    try:
        snapshots = []
        for row in connection.execute("SELECT * FROM popularity_snapshots ORDER BY id"):
            value = dict(row)
            payload_json = value.pop("payload_json")
            value["payload"] = json.loads(payload_json)
            value["payload_canonical_json"] = payload_json
            snapshots.append(value)
        scores = []
        for row in connection.execute("SELECT * FROM popularity_scores ORDER BY id"):
            value = dict(row)
            value["draw"] = json.loads(value.pop("draw_json"))
            value["provenance"] = json.loads(value.pop("provenance_json"))
            scores.append(value)
        ledger_summary = _verify_connection(connection)
    finally:
        connection.close()
    return {
        "format": POPULARITY_EVIDENCE_FORMAT,
        "schema_version": POPULARITY_EVIDENCE_SCHEMA_VERSION,
        "ledger": ledger_summary,
        "snapshots": snapshots,
        "scores": scores,
    }


def verify_popularity_evidence(evidence: object) -> dict[str, object]:
    errors: list[str] = []
    if not isinstance(evidence, dict):
        return {"valid": False, "errors": ["evidence:not_object"]}
    if evidence.get("format") != POPULARITY_EVIDENCE_FORMAT:
        errors.append("evidence:format")
    if evidence.get("schema_version") != POPULARITY_EVIDENCE_SCHEMA_VERSION:
        errors.append("evidence:schema_version")
    snapshots = evidence.get("snapshots")
    scores = evidence.get("scores")
    claimed = evidence.get("ledger")
    if not isinstance(snapshots, list) or not isinstance(scores, list):
        return {"valid": False, "errors": [*errors, "evidence:records"]}
    if not isinstance(claimed, dict):
        return {"valid": False, "errors": [*errors, "evidence:ledger"]}

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(POPULARITY_LEDGER_SCHEMA)
    try:
        for position, snapshot in enumerate(snapshots, start=1):
            try:
                row = dict(snapshot)
                payload = row.pop("payload")
                payload_json = row.pop("payload_canonical_json", None)
                if not isinstance(payload_json, str):
                    raise ValueError("Payload canonique manquant")
                if _canonical_json(json.loads(payload_json)) != _canonical_json(payload):
                    errors.append(f"snapshot:{position}:payload_mismatch")
                row["payload_json"] = payload_json
                columns = tuple(row)
                connection.execute(
                    f"INSERT INTO popularity_snapshots({','.join(columns)}) "
                    f"VALUES ({','.join('?' for _ in columns)})",
                    tuple(row.values()),
                )
            except (KeyError, TypeError, ValueError, sqlite3.Error, json.JSONDecodeError):
                errors.append(f"snapshot:{position}:invalid")
        for position, score in enumerate(scores, start=1):
            try:
                row = dict(score)
                row["draw_json"] = _canonical_json(row.pop("draw"))
                row["provenance_json"] = _canonical_json(row.pop("provenance"))
                columns = tuple(row)
                connection.execute(
                    f"INSERT INTO popularity_scores({','.join(columns)}) "
                    f"VALUES ({','.join('?' for _ in columns)})",
                    tuple(row.values()),
                )
            except (KeyError, TypeError, ValueError, sqlite3.Error):
                errors.append(f"score:{position}:invalid")
        computed = _verify_connection(connection)
        errors.extend(computed["errors"])
        if computed != claimed:
            errors.append("evidence:ledger_summary")
    finally:
        connection.close()
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "snapshots": len(snapshots),
        "scores": len(scores),
    }
