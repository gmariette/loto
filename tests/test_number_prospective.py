import copy
import math
import sqlite3
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from loto_lab.domain import Draw
from loto_lab.model_identity import build_number_model_specification
from loto_lab.number_prospective import (
    NUMBER_EVIDENCE_FORMAT,
    _exact_top5_tail,
    cohort_alpha_budget,
    export_number_evidence,
    record_number_forecast,
    score_pending_number_forecasts,
    verify_number_evidence,
    verify_number_ledger,
)
from loto_lab.prospective import build_data_provenance


class NumberProspectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.ledger = self.root / "numbers.sqlite"
        self.source = self.root / "draws.sqlite"
        self.source.write_bytes(b"number-test-data")
        self.base_draws = [Draw((1, 2, 3, 4, 5), 1, date(2028, 12, 31))]
        self.provenance = {"data": build_data_provenance([self.source], self.base_draws)}
        self.score_provenance = {
            **self.provenance,
            "result_source": "https://www.fdj.fr/jeux-de-tirage/loto/resultats",
        }
        self.specification = self._specification(seed=42)

    def tearDown(self) -> None:
        self.directory.cleanup()

    @staticmethod
    def _specification(seed: int) -> dict[str, object]:
        return build_number_model_specification(
            game="loto",
            min_history=50,
            min_train=500,
            outer_folds=3,
            simulations=2_000,
            seed=seed,
            models=("bayesian", "logistic"),
        )

    @staticmethod
    def _prediction(target: date) -> dict[str, object]:
        return {
            "status": "forced_experimental",
            "game": "loto",
            "target_date": target,
            "training_last_date": target - timedelta(days=1),
            "model": "logistic",
            "parameters": {"c": 0.01, "uniform_blend": 0.0},
            "numbers": (1, 2, 3, 4, 5),
            "chance": 1,
            "validation": {"qualified": False},
        }

    def test_append_only_record_score_and_export(self) -> None:
        target = date(2030, 1, 2)
        record = record_number_forecast(
            self.ledger,
            self._prediction(target),
            model_specification=self.specification,
            provenance=self.provenance,
            current_time=datetime(2030, 1, 1, tzinfo=UTC),
        )
        self.assertEqual(record["cohort_index"], 1)
        self.assertEqual(record["alpha_budget"], 0.025)
        with self.assertRaisesRegex(ValueError, "existe deja"):
            record_number_forecast(
                self.ledger,
                self._prediction(target),
                model_specification=self.specification,
                provenance=self.provenance,
                current_time=datetime(2030, 1, 1, tzinfo=UTC),
            )

        draw = Draw((1, 2, 6, 7, 8), 1, target)
        early = score_pending_number_forecasts(
            self.ledger,
            [draw],
            provenance=self.score_provenance,
            current_time=datetime(2030, 1, 2, 18, tzinfo=UTC),
        )
        self.assertEqual(early["new_scores"], 0)
        self.assertEqual(early["skipped"][0]["reason"], "tirage_non_cloture")
        scored = score_pending_number_forecasts(
            self.ledger,
            [draw],
            provenance=self.score_provenance,
            current_time=datetime(2030, 1, 2, 22, tzinfo=UTC),
        )
        self.assertEqual(scored["scores"][0]["main_hits"], 2)
        self.assertTrue(scored["scores"][0]["chance_hit"])
        self.assertTrue(verify_number_ledger(self.ledger)["valid"])
        evidence = export_number_evidence(self.ledger)
        self.assertEqual(evidence["format"], NUMBER_EVIDENCE_FORMAT)
        self.assertEqual(evidence["forecasts"][0]["main"], [1, 2, 3, 4, 5])
        self.assertTrue(verify_number_evidence(evidence)["valid"])
        tampered = copy.deepcopy(evidence)
        tampered["forecasts"][0]["main"][0] = 49
        self.assertFalse(verify_number_evidence(tampered)["valid"])
        tampered_summary = copy.deepcopy(evidence)
        tampered_summary["ledger"]["cohorts"][0]["qualification_status"] = "qualified"
        self.assertFalse(verify_number_evidence(tampered_summary)["valid"])

        connection = sqlite3.connect(self.ledger)
        try:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                connection.execute("UPDATE number_forecasts SET chance = 2 WHERE id = 1")
        finally:
            connection.close()

    def test_new_scientific_identity_spends_next_alpha_budget(self) -> None:
        first = record_number_forecast(
            self.ledger,
            self._prediction(date(2030, 1, 2)),
            model_specification=self.specification,
            provenance=self.provenance,
            current_time=datetime(2030, 1, 1, tzinfo=UTC),
        )
        second = record_number_forecast(
            self.ledger,
            self._prediction(date(2030, 1, 3)),
            model_specification=self._specification(seed=43),
            provenance=self.provenance,
            current_time=datetime(2030, 1, 1, tzinfo=UTC),
        )
        self.assertEqual(first["cohort_index"], 1)
        self.assertEqual(second["cohort_index"], 2)
        self.assertAlmostEqual(second["alpha_budget"], 0.05 / 6)
        self.assertLess(sum(cohort_alpha_budget(index) for index in range(1, 10_000)), 0.05)

    def test_exact_null_and_qualification_after_first_hundred_scores(self) -> None:
        start = date(2030, 1, 2)
        recording_time = datetime(2030, 1, 1, tzinfo=UTC)
        draws = []
        for offset in range(100):
            target = start + timedelta(days=offset)
            record_number_forecast(
                self.ledger,
                self._prediction(target),
                model_specification=self.specification,
                provenance=self.provenance,
                current_time=recording_time,
            )
            draws.append(Draw((1, 2, 3, 4, 5), 1, target))
        score_pending_number_forecasts(
            self.ledger,
            draws,
            provenance=self.score_provenance,
            current_time=datetime(2031, 1, 1, tzinfo=UTC),
        )
        cohort = verify_number_ledger(self.ledger)["cohorts"][0]
        self.assertEqual(cohort["qualification_scores"], 100)
        self.assertEqual(cohort["qualification_status"], "qualified")
        self.assertGreater(cohort["uplift_ci_low"], 0)
        self.assertLess(cohort["exact_p_value"], cohort["alpha_budget"])
        self.assertAlmostEqual(_exact_top5_tail(1, 5), 1 / math.comb(49, 5))

    def test_retroactive_prediction_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "retroactive"):
            record_number_forecast(
                self.ledger,
                self._prediction(date(2030, 1, 2)),
                model_specification=self.specification,
                provenance=self.provenance,
                current_time=datetime(2030, 1, 3, tzinfo=UTC),
            )


if __name__ == "__main__":
    unittest.main()
