from __future__ import annotations

import fcntl
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from .data import load_draws_many
from .domain import Draw
from .prospective import (
    GENESIS_HASH,
    _canonical_json,
    _digest,
    _recording_is_open,
    _validate_official_source,
    build_data_provenance,
    export_ledger_evidence,
    forecast_state,
    ledger_info,
    record_value_forecast,
    score_pending_forecasts,
)
from .value import TICKET_PRICES, ValueReport, estimate_value

MANIFEST_SCHEMA_VERSION = 1
JOURNAL_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ProspectiveManifest:
    source: Path
    manifest_hash: str
    enabled: bool
    data: tuple[Path, ...]
    ledger: Path
    evidence: Path
    journal: Path
    game: str
    target_date: date
    jackpot: float
    jackpot_source: str
    result_source: str | None
    simulations: int
    seed: int

    @classmethod
    def load(cls, path: str | Path) -> ProspectiveManifest:
        source = Path(path).resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Manifeste illisible: {error}") from error
        if not isinstance(raw, dict):
            raise ValueError("La racine du manifeste doit etre un objet JSON")
        allowed = {
            "schema_version",
            "enabled",
            "data",
            "ledger",
            "evidence",
            "journal",
            "game",
            "target_date",
            "jackpot",
            "jackpot_source",
            "result_source",
            "simulations",
            "seed",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Champs inconnus dans le manifeste: {', '.join(unknown)}")
        required = allowed - {"result_source", "simulations", "seed"}
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"Champs manquants dans le manifeste: {', '.join(missing)}")
        if raw["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError("Version de manifeste non prise en charge")
        if not isinstance(raw["enabled"], bool):
            raise ValueError("Le champ enabled doit etre un booleen")
        if not isinstance(raw["data"], list) or not raw["data"]:
            raise ValueError("Le manifeste doit contenir au moins un fichier de donnees")
        base = source.parent

        def resolve(value: object) -> Path:
            candidate = Path(str(value))
            return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

        data = tuple(resolve(value) for value in raw["data"])
        if any(not path.is_file() for path in data):
            raise ValueError("Tous les fichiers de donnees du manifeste doivent exister")
        game = str(raw["game"])
        if game not in TICKET_PRICES:
            raise ValueError(f"Jeu inconnu: {game}")
        try:
            target_date = date.fromisoformat(str(raw["target_date"]))
            jackpot = float(raw["jackpot"])
            simulations = int(raw.get("simulations", 2_000))
            seed = int(raw.get("seed", 0))
        except (TypeError, ValueError) as error:
            raise ValueError("Date, jackpot, simulations ou graine invalides") from error
        if jackpot <= 0 or simulations < 100:
            raise ValueError("Le jackpot doit etre positif et simulations >= 100")
        jackpot_source = _validate_official_source(raw["jackpot_source"])
        result_value = raw.get("result_source")
        result_source = (
            _validate_official_source(result_value) if result_value is not None else None
        )
        return cls(
            source=source,
            manifest_hash=_digest(raw),
            enabled=raw["enabled"],
            data=data,
            ledger=resolve(raw["ledger"]),
            evidence=resolve(raw["evidence"]),
            journal=resolve(raw["journal"]),
            game=game,
            target_date=target_date,
            jackpot=jackpot,
            jackpot_source=jackpot_source,
            result_source=result_source,
            simulations=simulations,
            seed=seed,
        )


def write_json_atomic(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, target.stat().st_mode & 0o777 if target.exists() else 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_journal_entries(lines: list[str]) -> dict[str, object]:
    errors = []
    previous_hash = GENESIS_HASH
    entries = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise TypeError
            record_hash = str(entry.pop("record_hash"))
            if entry.get("schema_version") != JOURNAL_SCHEMA_VERSION:
                errors.append(f"line:{line_number}:schema_version")
            if entry.get("previous_hash") != previous_hash or _digest(entry) != record_hash:
                errors.append(f"line:{line_number}:hash")
            previous_hash = record_hash
            entries += 1
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            errors.append(f"line:{line_number}:malformed")
    return {
        "valid": not errors,
        "errors": errors,
        "entries": entries,
        "head_hash": previous_hash,
    }


def verify_operation_journal(path: str | Path) -> dict[str, object]:
    target = Path(path)
    if not target.exists():
        return {"valid": True, "errors": [], "entries": 0, "head_hash": GENESIS_HASH}
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return {
            "valid": False,
            "errors": [f"read:{error}"],
            "entries": 0,
            "head_hash": GENESIS_HASH,
        }
    return _verify_journal_entries(lines)


def append_operation_journal(
    path: str | Path,
    *,
    manifest: ProspectiveManifest,
    action: str,
    status: str,
    details: dict[str, object],
    current_time: datetime,
) -> dict[str, object]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        verification = _verify_journal_entries(stream.read().splitlines())
        if not verification["valid"]:
            raise ValueError("Le journal d'operations est invalide; ajout refuse")
        entry: dict[str, object] = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "created_at": current_time.astimezone(UTC).isoformat(),
            "manifest_hash": manifest.manifest_hash,
            "game": manifest.game,
            "target_date": manifest.target_date.isoformat(),
            "action": action,
            "status": status,
            "details": details,
            "previous_hash": verification["head_hash"],
        }
        entry["record_hash"] = _digest(entry)
        stream.seek(0, os.SEEK_END)
        stream.write(_canonical_json(entry) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return entry


def _report_for_manifest(
    manifest: ProspectiveManifest, draws: list[Draw]
) -> tuple[ValueReport, dict[str, object]]:
    report = estimate_value(
        draws,
        jackpot=manifest.jackpot,
        game=manifest.game,
        expected_co_winners=None,
        target_date=manifest.target_date,
        bootstrap_simulations=manifest.simulations,
        seed=manifest.seed,
    )
    provenance = {
        "data": build_data_provenance(list(manifest.data), draws),
        "jackpot_source": manifest.jackpot_source,
        "bootstrap_simulations": manifest.simulations,
        "seed": manifest.seed,
        "manifest_hash": manifest.manifest_hash,
    }
    return report, provenance


def run_prospective_cycle(
    manifest_path: str | Path,
    *,
    dry_run: bool = False,
    current_time: datetime | None = None,
) -> dict[str, object]:
    manifest = ProspectiveManifest.load(manifest_path)
    if not manifest.enabled and not dry_run:
        raise ValueError("Le manifeste doit contenir enabled=true avant une execution reelle")
    now = current_time or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("L'heure d'execution doit inclure un fuseau horaire")
    draws = load_draws_many(list(manifest.data))
    state_before = forecast_state(
        manifest.ledger,
        manifest.game,
        manifest.target_date,
        migrate=not dry_run,
    )
    target_present = any(
        draw.game == manifest.game and draw.draw_date == manifest.target_date
        for draw in draws
    )
    recording_open = _recording_is_open(manifest.target_date, now)
    state = str(state_before["state"])
    action = "none"
    status = "unchanged"
    details: dict[str, object] = {}

    if state == "missing":
        if target_present:
            status = "missed_before_result_import"
        elif not recording_open:
            status = "missed_deadline"
        else:
            action = "record"
            report, provenance = _report_for_manifest(manifest, draws)
            details["projected_report"] = report.to_dict()
            if dry_run:
                status = "planned_record"
            else:
                details["record"] = record_value_forecast(
                    manifest.ledger, report, provenance=provenance
                )
                status = "recorded_waiting_result"
    elif state == "pending":
        if manifest.result_source is None or not target_present:
            status = "waiting_result"
        elif recording_open:
            status = "waiting_closure"
        else:
            action = "score"
            if dry_run:
                status = "planned_score"
            else:
                score_provenance = {
                    "data": build_data_provenance(list(manifest.data), draws),
                    "result_source": manifest.result_source,
                    "manifest_hash": manifest.manifest_hash,
                }
                details["scoring"] = score_pending_forecasts(
                    manifest.ledger,
                    draws,
                    provenance=score_provenance,
                    current_time=now,
                )
                status = "scored"
    elif state == "scored":
        status = "already_scored"
    else:
        raise ValueError(f"Etat prospectif inconnu: {state}")

    if not dry_run and manifest.ledger.exists():
        evidence = export_ledger_evidence(manifest.ledger)
        write_json_atomic(manifest.evidence, evidence)
        details["ledger"] = ledger_info(manifest.ledger)
        details["evidence"] = str(manifest.evidence)
    state_after = (
        forecast_state(manifest.ledger, manifest.game, manifest.target_date)
        if not dry_run and manifest.ledger.exists()
        else state_before
    )
    result = {
        "manifest": str(manifest.source),
        "manifest_hash": manifest.manifest_hash,
        "enabled": manifest.enabled,
        "dry_run": dry_run,
        "action": action,
        "status": status,
        "state_before": state_before,
        "state_after": state_after,
        "details": details,
    }
    if not dry_run:
        journal_entry = append_operation_journal(
            manifest.journal,
            manifest=manifest,
            action=action,
            status=status,
            details={
                "state_before": state_before,
                "state_after": state_after,
                "evidence": str(manifest.evidence) if manifest.ledger.exists() else None,
            },
            current_time=now,
        )
        result["journal_hash"] = journal_entry["record_hash"]
    return result
