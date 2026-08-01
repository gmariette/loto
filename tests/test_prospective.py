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
    _recording_is_open,
    ledger_info,
    record_value_forecast,
    score_pending_forecasts,
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
        scoring = score_pending_forecasts(self.ledger, [*self.draws, target])
        self.assertEqual(scoring["new_scores"], 1)
        self.assertEqual(score_pending_forecasts(self.ledger, [target])["new_scores"], 0)

        info = ledger_info(self.ledger)
        self.assertTrue(info["valid"])
        self.assertEqual(info["forecasts"], 1)
        self.assertEqual(info["scores"], 1)
        self.assertEqual(info["pending"], 0)
        self.assertIsNotNone(info["mae"])

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
