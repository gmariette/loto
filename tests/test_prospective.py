import copy
import hashlib
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
    _benchmark_cohort_summary,
    _canonical_json,
    _recording_is_open,
    _score_hash,
    build_data_provenance,
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
        self.source = Path(self.directory.name) / "draws.sqlite"
        self.source.write_bytes(b"test-draws")
        self.data_provenance = build_data_provenance([self.source], self.draws)
        self.forecast_provenance = {
            "jackpot_source": "https://www.fdj.fr/jeux-de-tirage/loto/resultats",
            "data": self.data_provenance,
        }
        self.score_provenance = {
            "result_source": "https://www.fdj.fr/jeux-de-tirage/loto/resultats",
            "data": self.data_provenance,
        }

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_append_only_forecast_and_score_chain(self) -> None:
        record = record_value_forecast(
            self.ledger, self.report, provenance=self.forecast_provenance
        )
        self.assertEqual(record["target_date"], self.target_date.isoformat())
        self.assertTrue(verify_ledger(self.ledger)["valid"])

        with self.assertRaisesRegex(ValueError, "existe deja"):
            record_value_forecast(
                self.ledger, self.report, provenance=self.forecast_provenance
            )

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
        early = score_pending_forecasts(
            self.ledger,
            [*self.draws, target],
            provenance=self.score_provenance,
        )
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
            self.ledger,
            [*self.draws, target],
            provenance=self.score_provenance,
            current_time=scoring_time,
        )
        self.assertEqual(scoring["new_scores"], 1)
        self.assertEqual(
            score_pending_forecasts(
                self.ledger,
                [target],
                provenance=self.score_provenance,
                current_time=scoring_time,
            )["new_scores"],
            0,
        )

        info = ledger_info(self.ledger)
        self.assertTrue(info["valid"])
        self.assertEqual(info["forecasts"], 1)
        self.assertEqual(info["scores"], 1)
        self.assertEqual(info["pending"], 0)
        self.assertIsNotNone(info["mae"])
        self.assertEqual(info["benchmark_observations"], 1)
        self.assertEqual(
            info["qualification_cohorts"][0]["qualification_status"],
            "insufficient_data",
        )

        evidence = export_ledger_evidence(self.ledger)
        self.assertTrue(verify_evidence(evidence)["valid"])

        legacy_evidence = copy.deepcopy(evidence)
        legacy_evidence["schema_version"] = 1
        legacy_score = legacy_evidence["scores"][0]
        legacy_score.pop("provenance")
        legacy_score.pop("hash_version")
        legacy_score.pop("metrics_version")
        for key in (
            "naive_ev",
            "naive_error",
            "naive_absolute_error",
            "absolute_error_delta",
        ):
            legacy_score.pop(key)
        legacy_metrics = {
            key: legacy_score[key]
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
        legacy_score["record_hash"] = _score_hash(
            legacy_score["scored_at"],
            legacy_evidence["forecasts"][0]["record_hash"],
            legacy_score["previous_hash"],
            _canonical_json(legacy_score["draw"]),
            legacy_metrics,
        )
        legacy_evidence["ledger"]["score_head_hash"] = legacy_score["record_hash"]
        self.assertTrue(verify_evidence(legacy_evidence)["valid"])

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
                "metrics_version",
                "naive_ev",
                "naive_error",
                "naive_absolute_error",
                "absolute_error_delta",
            )
        }
        score["record_hash"] = _score_hash(
            score["scored_at"],
            tampered_metrics["forecasts"][0]["record_hash"],
            score["previous_hash"],
            _canonical_json(score["draw"]),
            metrics,
            provenance_json=_canonical_json(score["provenance"]),
            hash_version=score["hash_version"],
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

    def test_data_provenance_fingerprints_file_and_loaded_draws(self) -> None:
        source = Path(self.directory.name) / "archive.bin"
        source.write_bytes(b"archive-test")
        provenance = build_data_provenance([source], self.draws)
        self.assertEqual(
            provenance["files"][0]["sha256"],
            hashlib.sha256(b"archive-test").hexdigest(),
        )
        self.assertEqual(provenance["draws_snapshot"]["count"], len(self.draws))
        self.assertEqual(len(provenance["draws_snapshot"]["sha256"]), 64)

    def test_v1_ledger_is_migrated_without_records_being_rewritten(self) -> None:
        record = record_value_forecast(
            self.ledger, self.report, provenance=self.forecast_provenance
        )
        connection = sqlite3.connect(self.ledger)
        try:
            connection.execute("ALTER TABLE value_scores DROP COLUMN score_provenance_json")
            connection.execute("ALTER TABLE value_scores DROP COLUMN hash_version")
            connection.execute("ALTER TABLE value_scores DROP COLUMN metrics_version")
            connection.execute("ALTER TABLE value_scores DROP COLUMN absolute_error_delta")
            connection.execute("ALTER TABLE value_scores DROP COLUMN naive_absolute_error")
            connection.execute("ALTER TABLE value_scores DROP COLUMN naive_error")
            connection.execute("ALTER TABLE value_scores DROP COLUMN naive_ev")
            connection.execute(
                "UPDATE ledger_metadata SET value = '1' WHERE key = 'schema_version'"
            )
            connection.commit()
        finally:
            connection.close()

        self.assertTrue(ledger_info(self.ledger)["valid"])
        connection = sqlite3.connect(self.ledger)
        try:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(value_scores)")
            }
            schema_version = connection.execute(
                "SELECT value FROM ledger_metadata WHERE key = 'schema_version'"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertIn("score_provenance_json", columns)
        self.assertIn("hash_version", columns)
        self.assertIn("metrics_version", columns)
        self.assertIn("absolute_error_delta", columns)
        self.assertEqual(schema_version, "3")
        self.assertEqual(verify_ledger(self.ledger)["forecast_head_hash"], record["record_hash"])

    def test_v2_score_hash_survives_v3_migration(self) -> None:
        forecast = record_value_forecast(
            self.ledger, self.report, provenance=self.forecast_provenance
        )
        source_draw = self.draws[-1]
        target = Draw(
            source_draw.main,
            source_draw.chance,
            self.target_date,
            source_draw.game,
            source_draw.prizes,
            10,
            20_000,
        )
        scoring_time = datetime(
            self.target_date.year,
            self.target_date.month,
            self.target_date.day,
            21,
            tzinfo=PARIS_TIMEZONE,
        )
        score_pending_forecasts(
            self.ledger,
            [target],
            provenance=self.score_provenance,
            current_time=scoring_time,
        )

        connection = sqlite3.connect(self.ledger)
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute("SELECT * FROM value_scores WHERE id = 1").fetchone()
            legacy_metrics = {
                key: row[key]
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
            legacy_hash = _score_hash(
                row["scored_at"],
                forecast["record_hash"],
                row["previous_hash"],
                row["draw_json"],
                legacy_metrics,
                provenance_json=row["score_provenance_json"],
                hash_version=2,
            )
            connection.execute("DROP TRIGGER value_scores_no_update")
            connection.execute(
                """
                UPDATE value_scores
                SET naive_ev = NULL, naive_error = NULL, naive_absolute_error = NULL,
                    absolute_error_delta = NULL, metrics_version = 1, record_hash = ?
                WHERE id = 1
                """,
                (legacy_hash,),
            )
            connection.execute("ALTER TABLE value_scores DROP COLUMN metrics_version")
            connection.execute("ALTER TABLE value_scores DROP COLUMN absolute_error_delta")
            connection.execute("ALTER TABLE value_scores DROP COLUMN naive_absolute_error")
            connection.execute("ALTER TABLE value_scores DROP COLUMN naive_error")
            connection.execute("ALTER TABLE value_scores DROP COLUMN naive_ev")
            connection.execute(
                "UPDATE ledger_metadata SET value = '2' WHERE key = 'schema_version'"
            )
            connection.commit()
        finally:
            connection.close()

        info = ledger_info(self.ledger)
        self.assertTrue(info["valid"])
        self.assertEqual(info["score_head_hash"], legacy_hash)
        self.assertEqual(info["benchmark_observations"], 0)

    def test_score_requires_an_official_fdj_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "officielle FDJ"):
            score_pending_forecasts(
                self.ledger,
                self.draws,
                provenance={"result_source": "https://example.com/resultat"},
            )

    def test_prospective_qualification_is_frozen_on_first_100_scores(self) -> None:
        first_cohort = [
            {
                "absolute_error": 0.1,
                "naive_absolute_error": 0.2,
                "absolute_error_delta": -0.1,
                "covered": int(index < 95),
            }
            for index in range(100)
        ]
        insufficient = _benchmark_cohort_summary("0.10.0", first_cohort[:99])
        self.assertEqual(insufficient["qualification_status"], "insufficient_data")

        qualified = _benchmark_cohort_summary("0.10.0", first_cohort)
        self.assertEqual(qualified["qualification_status"], "qualified")
        later_scores = [
            {
                "absolute_error": 1.0,
                "naive_absolute_error": 0.1,
                "absolute_error_delta": 0.9,
                "covered": 0,
            }
            for _ in range(20)
        ]
        frozen = _benchmark_cohort_summary("0.10.0", [*first_cohort, *later_scores])
        self.assertEqual(frozen["qualification_status"], "qualified")
        self.assertEqual(frozen["evaluation_observations"], 100)
        self.assertLess(frozen["evaluation_mae_delta"], 0)
        self.assertGreater(frozen["monitoring_mae_delta"], 0)

    def test_retroactive_forecast_is_rejected(self) -> None:
        retroactive = replace(self.report, target_date=date.today() - timedelta(days=1))
        with self.assertRaisesRegex(ValueError, "retroactive"):
            record_value_forecast(
                self.ledger, retroactive, provenance=self.forecast_provenance
            )

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
