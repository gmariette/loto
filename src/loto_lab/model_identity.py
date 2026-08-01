from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

MODEL_SPECIFICATION_FORMAT = "loto-lab.value-model"
MODEL_SPECIFICATION_VERSION = 1
NUMBER_MODEL_SPECIFICATION_FORMAT = "loto-lab.number-model"
NUMBER_MODEL_SPECIFICATION_VERSION = 1
MODEL_SOURCE_FILES = (
    "domain.py",
    "participation.py",
    "probability.py",
    "value.py",
)
NUMBER_MODEL_SOURCE_FILES = (
    "domain.py",
    "ml.py",
    "number_prospective.py",
    "popularity.py",
    "probability.py",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _source_fingerprints(
    names: tuple[str, ...] = MODEL_SOURCE_FILES,
) -> list[dict[str, str]]:
    package = Path(__file__).parent
    return [
        {
            "path": f"loto_lab/{name}",
            "sha256": hashlib.sha256((package / name).read_bytes()).hexdigest(),
        }
        for name in names
    ]


def build_model_specification(
    *,
    game: str,
    bootstrap_simulations: int,
    seed: int,
) -> dict[str, object]:
    specification: dict[str, object] = {
        "format": MODEL_SPECIFICATION_FORMAT,
        "specification_version": MODEL_SPECIFICATION_VERSION,
        "source_files": _source_fingerprints(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
        },
        "parameters": {
            "game": game,
            "expected_co_winners": None,
            "popularity_factor": 1.0,
            "participation_min_train": 500,
            "participation_folds": 3,
            "bootstrap_simulations": bootstrap_simulations,
            "seed": seed,
        },
    }
    return {**specification, "sha256": _digest(specification)}


def build_number_model_specification(
    *,
    game: str,
    min_history: int,
    min_train: int,
    outer_folds: int,
    simulations: int,
    seed: int,
    models: tuple[str, ...],
    value_aware: bool = False,
    max_expected_hit_loss: float = 0.005,
    popularity_block_size: int = 12,
) -> dict[str, object]:
    specification: dict[str, object] = {
        "format": NUMBER_MODEL_SPECIFICATION_FORMAT,
        "specification_version": NUMBER_MODEL_SPECIFICATION_VERSION,
        "source_files": _source_fingerprints(NUMBER_MODEL_SOURCE_FILES),
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
        },
        "parameters": {
            "game": game,
            "min_history": min_history,
            "min_train": min_train,
            "outer_folds": outer_folds,
            "simulations": simulations,
            "seed": seed,
            "models": list(models),
            "value_aware": value_aware,
            "max_expected_hit_loss": max_expected_hit_loss,
            "popularity_block_size": popularity_block_size,
        },
    }
    return {**specification, "sha256": _digest(specification)}


def validate_model_specification(value: object) -> str:
    return _validate_model_specification(
        value, MODEL_SPECIFICATION_FORMAT, MODEL_SPECIFICATION_VERSION
    )


def validate_number_model_specification(value: object) -> str:
    return _validate_model_specification(
        value, NUMBER_MODEL_SPECIFICATION_FORMAT, NUMBER_MODEL_SPECIFICATION_VERSION
    )


def _validate_model_specification(
    value: object, expected_format: str, expected_version: int
) -> str:
    if not isinstance(value, dict):
        raise ValueError("La specification du modele doit etre un objet JSON")
    if value.get("format") != expected_format:
        raise ValueError("Format de specification du modele inconnu")
    if value.get("specification_version") != expected_version:
        raise ValueError("Version de specification du modele inconnue")
    claimed_hash = value.get("sha256")
    if (
        not isinstance(claimed_hash, str)
        or len(claimed_hash) != 64
        or any(character not in "0123456789abcdef" for character in claimed_hash)
    ):
        raise ValueError("Empreinte de specification du modele invalide")
    if not isinstance(value.get("source_files"), list):
        raise ValueError("Sources de specification du modele invalides")
    if not isinstance(value.get("runtime"), dict):
        raise ValueError("Environnement de specification du modele invalide")
    if not isinstance(value.get("parameters"), dict):
        raise ValueError("Parametres de specification du modele invalides")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if _digest(payload) != claimed_hash:
        raise ValueError("La specification du modele a ete modifiee")
    return claimed_hash
