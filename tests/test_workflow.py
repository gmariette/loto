import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from test_participation import participation_draws

from loto_lab.domain import Draw
from loto_lab.prospective import PARIS_TIMEZONE, ledger_info
from loto_lab.workflow import (
    ProspectiveManifest,
    run_prospective_cycle,
    verify_operation_journal,
)


class ProspectiveWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.data = self.root / "draws.sqlite"
        self.data.write_bytes(b"workflow-data")
        self.manifest = self.root / "manifest.json"
        self.ledger = self.root / "prospective.sqlite"
        self.evidence = self.root / "evidence.json"
        self.journal = self.root / "operations.jsonl"
        self.draws = participation_draws(600)
        self.target_date = date.today() + timedelta(days=2)
        self.payload = {
            "schema_version": 1,
            "enabled": True,
            "data": [self.data.name],
            "ledger": self.ledger.name,
            "evidence": self.evidence.name,
            "journal": self.journal.name,
            "game": "loto",
            "target_date": self.target_date.isoformat(),
            "jackpot": 5_000_000,
            "jackpot_source": "https://www.fdj.fr/jeux-de-tirage/loto/resultats",
            "simulations": 100,
            "seed": 4,
        }
        self._write_manifest()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _write_manifest(self) -> None:
        self.manifest.write_text(json.dumps(self.payload), encoding="utf-8")

    def test_dry_run_does_not_modify_files(self) -> None:
        with patch("loto_lab.workflow.load_draws_many", return_value=self.draws):
            result = run_prospective_cycle(self.manifest, dry_run=True)
        self.assertEqual(result["status"], "planned_record")
        self.assertFalse(self.ledger.exists())
        self.assertFalse(self.evidence.exists())
        self.assertFalse(self.journal.exists())

    def test_cycle_records_recovers_scores_and_is_idempotent(self) -> None:
        with patch("loto_lab.workflow.load_draws_many", return_value=self.draws):
            first = run_prospective_cycle(self.manifest)
            self.evidence.unlink()
            second = run_prospective_cycle(self.manifest)
        self.assertEqual(first["status"], "recorded_waiting_result")
        self.assertEqual(second["status"], "waiting_result")
        self.assertEqual(ledger_info(self.ledger)["forecasts"], 1)
        self.assertTrue(self.evidence.exists())
        self.assertEqual(verify_operation_journal(self.journal)["entries"], 2)

        source = self.draws[-1]
        target = Draw(
            source.main,
            source.chance,
            self.target_date,
            source.game,
            source.prizes,
            10,
            20_000,
        )
        self.payload["result_source"] = (
            "https://www.fdj.fr/jeux-de-tirage/loto/resultats/test"
        )
        self._write_manifest()
        scoring_time = datetime(
            self.target_date.year,
            self.target_date.month,
            self.target_date.day,
            21,
            tzinfo=PARIS_TIMEZONE,
        )
        with patch(
            "loto_lab.workflow.load_draws_many", return_value=[*self.draws, target]
        ):
            scored = run_prospective_cycle(
                self.manifest, current_time=scoring_time
            )
            repeated = run_prospective_cycle(
                self.manifest, current_time=scoring_time
            )
        self.assertEqual(scored["status"], "scored")
        self.assertEqual(repeated["status"], "already_scored")
        info = ledger_info(self.ledger)
        self.assertEqual(info["scores"], 1)
        self.assertEqual(info["benchmark_observations"], 1)
        journal = verify_operation_journal(self.journal)
        self.assertTrue(journal["valid"])
        self.assertEqual(journal["entries"], 4)

        content = self.journal.read_text(encoding="utf-8")
        self.journal.write_text(content.replace('"status":"scored"', '"status":"edited"'))
        self.assertFalse(verify_operation_journal(self.journal)["valid"])

    def test_manifest_rejects_unknown_fields(self) -> None:
        self.payload["unexpected"] = True
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "Champs inconnus"):
            ProspectiveManifest.load(self.manifest)

    def test_disabled_manifest_allows_only_dry_run(self) -> None:
        self.payload["enabled"] = False
        self._write_manifest()
        with patch("loto_lab.workflow.load_draws_many", return_value=self.draws):
            preview = run_prospective_cycle(self.manifest, dry_run=True)
        self.assertEqual(preview["status"], "planned_record")
        with self.assertRaisesRegex(ValueError, "enabled=true"):
            run_prospective_cycle(self.manifest)


if __name__ == "__main__":
    unittest.main()
