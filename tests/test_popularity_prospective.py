import copy
import random
import sqlite3
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from loto_lab.domain import Draw, PrizeResult, Ticket
from loto_lab.model_identity import build_popularity_model_specification
from loto_lab.popularity import (
    NUMBER_EFFECT_FEATURE_NAMES,
    _feature_matrix,
    fit_popularity_predictor,
    popularity_backtest,
)
from loto_lab.popularity_prospective import (
    POPULARITY_EVIDENCE_FORMAT,
    export_popularity_evidence,
    popularity_cohort_alpha_budget,
    record_popularity_snapshot,
    score_pending_popularity_snapshots,
    serialize_popularity_predictor,
    verify_popularity_evidence,
    verify_popularity_ledger,
)
from loto_lab.prospective import build_data_provenance


def popularity_draws(count: int = 240) -> list[Draw]:
    rng = random.Random(44)
    start = date(2020, 1, 1)
    draws = []
    for index in range(count):
        main = tuple(rng.sample(range(1, 50), 5))
        winners = 2 if all(number <= 31 for number in main) else 0
        draws.append(
            Draw(
                main,
                rng.randint(1, 10),
                start + timedelta(days=index),
                prizes=(
                    PrizeResult(1, winners, 2_000_000.0),
                    PrizeResult(2, 3 if winners else 0, 100_000.0),
                    PrizeResult(9, 100_000, 2.2),
                ),
            )
        )
    return draws


class PopularityProspectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.ledger = self.root / "popularity.sqlite"
        self.source = self.root / "draws.sqlite"
        self.source.write_bytes(b"popularity-test-data")
        self.draws = popularity_draws()
        self.provenance = {"data": build_data_provenance([self.source], self.draws)}
        self.score_provenance = {
            **self.provenance,
            "result_source": "https://www.fdj.fr/jeux-de-tirage/loto/resultats",
        }
        self.result = popularity_backtest(
            self.draws,
            min_train=100,
            outer_folds=2,
            simulations=200,
            block_size=6,
            seed=12,
        )
        self.predictor = fit_popularity_predictor(
            self.draws, self.result, bootstrap_models=20, seed=12
        )
        self.specification = self._specification(12)

    def tearDown(self) -> None:
        self.directory.cleanup()

    @staticmethod
    def _specification(
        seed: int, target: str = "jackpot"
    ) -> dict[str, object]:
        return build_popularity_model_specification(
            game="loto",
            min_train=100,
            outer_folds=2,
            simulations=200,
            block_size=6,
            seed=seed,
            bootstrap_models=20,
            uncertainty_quantile=0.9,
            target=target,
        )

    @staticmethod
    def _result_draw(target: date, main: tuple[int, ...] = (1, 2, 3, 4, 5)) -> Draw:
        return Draw(
            main,
            1,
            target,
            prizes=(
                PrizeResult(1, 2, 2_000_000.0),
                PrizeResult(2, 3, 100_000.0),
                PrizeResult(9, 100_000, 2.2),
            ),
        )

    def test_serialized_model_reproduces_point_prediction(self) -> None:
        model = serialize_popularity_predictor(self.predictor)
        draw = self._result_draw(date(2030, 1, 1), (32, 33, 34, 35, 36))
        ordered = np.sort(np.asarray([draw.main]), axis=1)
        features = np.column_stack(
            (
                np.sum(ordered > 31, axis=1),
                np.all(ordered <= 31, axis=1),
                np.sum(np.diff(ordered, axis=1) == 1, axis=1),
                np.sum(np.isin(ordered, (7, 13)), axis=1),
                np.sum(ordered, axis=1) / 150,
                np.abs(np.sum(ordered, axis=1) - 125) / 100,
            )
        )
        restored = np.exp(
            model["raw_intercept"]
            + features @ np.asarray(model["raw_coefficients"], dtype=float)
        )[0]
        self.assertAlmostEqual(restored, self.predictor.multiplier(Ticket(draw.main, 1)))

    def test_number_effect_model_serialization_reproduces_prediction(self) -> None:
        result = popularity_backtest(
            self.draws,
            min_train=100,
            outer_folds=2,
            simulations=200,
            block_size=6,
            seed=12,
            feature_set="number_effects",
        )
        predictor = fit_popularity_predictor(
            self.draws, result, bootstrap_models=20, seed=12
        )
        model = serialize_popularity_predictor(predictor)
        draw = self._result_draw(date(2030, 1, 1), (1, 7, 32, 41, 49))
        features = _feature_matrix(
            np.asarray([draw.main]),
            np.asarray([draw.chance]),
            NUMBER_EFFECT_FEATURE_NAMES,
        )
        restored = np.exp(
            model["raw_intercept"]
            + features @ np.asarray(model["raw_coefficients"], dtype=float)
        )[0]
        self.assertEqual(tuple(model["feature_names"]), NUMBER_EFFECT_FEATURE_NAMES)
        self.assertAlmostEqual(restored, predictor.multiplier(Ticket(draw.main, 1)))

    def test_append_only_record_score_export_and_tamper_detection(self) -> None:
        target = self.draws[-1].draw_date + timedelta(days=1)
        record = record_popularity_snapshot(
            self.ledger,
            self.predictor,
            self.draws,
            target,
            model_specification=self.specification,
            provenance=self.provenance,
            current_time=datetime(2020, 1, 1, tzinfo=UTC),
        )
        self.assertEqual(record["cohort_index"], 1)
        self.assertEqual(record["alpha_budget"], 0.025)
        with self.assertRaisesRegex(ValueError, "existe deja"):
            record_popularity_snapshot(
                self.ledger,
                self.predictor,
                self.draws,
                target,
                model_specification=self.specification,
                provenance=self.provenance,
                current_time=datetime(2020, 1, 1, tzinfo=UTC),
            )

        draw = self._result_draw(target)
        early = score_pending_popularity_snapshots(
            self.ledger,
            [draw],
            provenance=self.score_provenance,
            current_time=datetime.combine(target, datetime.min.time(), UTC),
        )
        self.assertEqual(early["new_scores"], 0)
        scored = score_pending_popularity_snapshots(
            self.ledger,
            [draw],
            provenance=self.score_provenance,
            current_time=datetime.combine(target + timedelta(days=1), datetime.min.time(), UTC),
        )
        self.assertEqual(scored["new_scores"], 1)
        self.assertGreater(scored["scores"][0]["predicted_winners"], 0)
        self.assertTrue(verify_popularity_ledger(self.ledger)["valid"])

        evidence = export_popularity_evidence(self.ledger)
        self.assertEqual(evidence["format"], POPULARITY_EVIDENCE_FORMAT)
        self.assertTrue(verify_popularity_evidence(evidence)["valid"])
        tampered = copy.deepcopy(evidence)
        tampered["snapshots"][0]["payload"]["model"]["raw_intercept"] += 1
        self.assertFalse(verify_popularity_evidence(tampered)["valid"])
        tampered_score = copy.deepcopy(evidence)
        tampered_score["scores"][0]["jackpot_winners"] = 3
        self.assertFalse(verify_popularity_evidence(tampered_score)["valid"])

        connection = sqlite3.connect(self.ledger)
        try:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                connection.execute(
                    "UPDATE popularity_snapshots SET game = 'super_loto' WHERE id = 1"
                )
        finally:
            connection.close()

    def test_new_identity_spends_next_alpha_budget_and_retroactive_is_rejected(self) -> None:
        target = self.draws[-1].draw_date + timedelta(days=1)
        first = record_popularity_snapshot(
            self.ledger,
            self.predictor,
            self.draws,
            target,
            model_specification=self.specification,
            provenance=self.provenance,
            current_time=datetime(2020, 1, 1, tzinfo=UTC),
        )
        second = record_popularity_snapshot(
            self.ledger,
            self.predictor,
            self.draws,
            target,
            model_specification=self._specification(13),
            provenance=self.provenance,
            current_time=datetime(2020, 1, 1, tzinfo=UTC),
        )
        self.assertEqual(first["cohort_index"], 1)
        self.assertEqual(second["cohort_index"], 2)
        self.assertAlmostEqual(second["alpha_budget"], 0.05 / 6)
        self.assertLess(
            sum(popularity_cohort_alpha_budget(index) for index in range(1, 10_000)), 0.05
        )
        with self.assertRaisesRegex(ValueError, "retroactivement"):
            record_popularity_snapshot(
                self.root / "retroactive.sqlite",
                self.predictor,
                self.draws,
                target,
                model_specification=self.specification,
                provenance=self.provenance,
                current_time=datetime(2030, 1, 1, tzinfo=UTC),
            )

    def test_main_combination_snapshot_scores_ranks_one_and_two(self) -> None:
        result = popularity_backtest(
            self.draws,
            target="main_combination",
            min_train=100,
            outer_folds=2,
            simulations=200,
            block_size=6,
            seed=12,
        )
        predictor = fit_popularity_predictor(
            self.draws, result, bootstrap_models=20, seed=12
        )
        target = self.draws[-1].draw_date + timedelta(days=1)
        record_popularity_snapshot(
            self.ledger,
            predictor,
            self.draws,
            target,
            model_specification=self._specification(12, "main_combination"),
            provenance=self.provenance,
            current_time=datetime(2020, 1, 1, tzinfo=UTC),
        )
        scored = score_pending_popularity_snapshots(
            self.ledger,
            [self._result_draw(target)],
            provenance=self.score_provenance,
            current_time=datetime(2030, 1, 1, tzinfo=UTC),
        )
        self.assertEqual(scored["scores"][0]["jackpot_winners"], 5)
        self.assertTrue(verify_popularity_ledger(self.ledger)["valid"])

    def test_qualification_is_frozen_on_first_hundred_scores(self) -> None:
        start = self.draws[-1].draw_date + timedelta(days=1)
        results = []
        for offset in range(100):
            target = start + timedelta(days=offset)
            record_popularity_snapshot(
                self.ledger,
                self.predictor,
                self.draws,
                target,
                model_specification=self.specification,
                provenance=self.provenance,
                current_time=datetime(2020, 1, 1, tzinfo=UTC),
            )
            results.append(self._result_draw(target))
        score_pending_popularity_snapshots(
            self.ledger,
            results,
            provenance=self.score_provenance,
            current_time=datetime(2030, 1, 1, tzinfo=UTC),
        )
        cohort = verify_popularity_ledger(self.ledger)["cohorts"][0]
        self.assertEqual(cohort["qualification_scores"], 100)
        self.assertEqual(cohort["qualification_status"], "qualified")
        self.assertLess(cohort["delta_ci_high"], 0)
        self.assertLess(cohort["block_sign_p_value"], cohort["alpha_budget"])


if __name__ == "__main__":
    unittest.main()
