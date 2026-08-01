import copy
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

from test_participation import participation_draws

from loto_lab.domain import Draw
from loto_lab.prospective import (
    PARIS_TIMEZONE,
    _canonical_json,
    _recording_is_open,
    _score_hash,
    export_ledger_evidence,
    ledger_info,
    record_value_forecast,
    score_pending_forecasts,
    verify_evidence,
    verify_ledger,
)
from loto_lab.value import estimate_value


class ProspectiveLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.ledger = Path(self.directory.name) / "prospective.sqlite"
        self.draws = participation_draws(180)
        self.target_date = date.today() + timedelta(days=2)
        self.report = estimate_value(
            self.draws,
            jackpot=5_000_000,
            target_date=self.target_date,
            bootstrap_simulations=100,
            seed=3,
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_append_only_forecast_and_score_chain(self) -> None:
        record = record_value_forecast(self.ledger, self.report)
        self.assertEqual(record["target_date"], self.target_date.isoformat())
        self.assertTrue(verify_ledger(self.ledger)["valid"])

        with self.assertRaisesRegex(ValueError, "existe deja"):
            record_value_forecast(self.ledger, self.report)

        target = self.draws[-1]
        target = Draw(
            target.main,
            target.chance,
            self.target_date,
            target.game,
            target.prizes,
            10,
            20_000,
        )
        early = score_pending_forecasts(self.ledger, [*self.draws, target])
        self.assertEqual(early["new_scores"], 0)
        self.assertEqual(early["skipped"][0]["reason"], "tirage_non_cloture")
        scoring_time = datetime(
            self.target_date.year,
            self.target_date.month,
            self.target_date.day,
            21,
            tzinfo=PARIS_TIMEZONE,
        )
        scoring = score_pending_forecasts(
            self.ledger, [*self.draws, target], current_time=scoring_time
        )
        self.assertEqual(scoring["new_scores"], 1)
        self.assertEqual(
            score_pending_forecasts(
                self.ledger, [target], current_time=scoring_time
            )["new_scores"],
            0,
        )

        info = ledger_info(self.ledger)
        self.assertTrue(info["valid"])
        self.assertEqual(info["forecasts"], 1)
        self.assertEqual(info["scores"], 1)
        self.assertEqual(info["pending"], 0)
        self.assertIsNotNone(info["mae"])

        evidence = export_ledger_evidence(self.ledger)
        self.assertTrue(verify_evidence(evidence)["valid"])

        tampered_forecast = copy.deepcopy(evidence)
        tampered_forecast["forecasts"][0]["jackpot"] = 1
        self.assertFalse(verify_evidence(tampered_forecast)["valid"])

        tampered_metrics = copy.deepcopy(evidence)
        score = tampered_metrics["scores"][0]
        score["observed_schedule_ev"] += 1
        metrics = {
            key: score[key]
            for key in (
                "observed_schedule_ev",
                "error",
                "absolute_error",
                "covered",
                "observed_positive",
                "false_positive",
                "false_negative",
            )
        }
        score["record_hash"] = _score_hash(
            score["scored_at"],
            tampered_metrics["forecasts"][0]["record_hash"],
            score["previous_hash"],
            _canonical_json(score["draw"]),
            metrics,
        )
        tampered_metrics["ledger"]["score_head_hash"] = score["record_hash"]
        verification = verify_evidence(tampered_metrics)
        self.assertFalse(verification["valid"])
        self.assertIn("score:1:metric:observed_schedule_ev", verification["errors"])

        connection = sqlite3.connect(self.ledger)
        try:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                connection.execute("UPDATE value_forecasts SET jackpot = 1 WHERE id = 1")
        finally:
            connection.close()

        connection = sqlite3.connect(self.ledger)
        try:
            connection.execute("DROP TRIGGER value_forecasts_no_update")
            connection.execute("UPDATE value_forecasts SET decision = 'eligible' WHERE id = 1")
            connection.commit()
        finally:
            connection.close()
        self.assertFalse(verify_ledger(self.ledger)["valid"])

    def test_retroactive_forecast_is_rejected(self) -> None:
        retroactive = replace(self.report, target_date=date.today() - timedelta(days=1))
        with self.assertRaisesRegex(ValueError, "retroactive"):
            record_value_forecast(self.ledger, retroactive)

    def test_same_day_recording_closes_at_french_deadline(self) -> None:
        target = date(2026, 8, 1)
        self.assertTrue(
            _recording_is_open(
                target, datetime(2026, 8, 1, 20, 14, tzinfo=PARIS_TIMEZONE)
            )
        )
        self.assertFalse(
            _recording_is_open(
                target, datetime(2026, 8, 1, 20, 15, tzinfo=PARIS_TIMEZONE)
            )
        )


if __name__ == "__main__":
    unittest.main()
