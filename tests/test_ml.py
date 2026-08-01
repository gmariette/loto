import random
import unittest
from datetime import date, timedelta

import numpy as np

from loto_lab.domain import Draw
from loto_lab.ml import (
    _fit_predict,
    blend_with_uniform,
    build_feature_dataset,
    nested_ml_backtest,
    next_feature_matrix,
    predict_next_draw,
    project_inclusion_probabilities,
)


def dated_random_draws(count: int, seed: int = 4) -> list[Draw]:
    rng = random.Random(seed)
    start = date(2020, 1, 1)
    return [
        Draw(
            tuple(rng.sample(range(1, 50), 5)),
            rng.randint(1, 10),
            start + timedelta(days=index),
        )
        for index in range(count)
    ]


class MLTests(unittest.TestCase):
    def test_projection_sums_to_five(self) -> None:
        projected = project_inclusion_probabilities(np.linspace(0.01, 0.8, 49))
        self.assertAlmostEqual(float(projected.sum()), 5)
        self.assertTrue(np.all((projected > 0) & (projected < 1)))

    def test_zero_blend_is_exactly_uniform(self) -> None:
        probabilities = np.linspace(0.01, 0.2, 49)[np.newaxis, :]
        blended = blend_with_uniform(probabilities, 0.0)
        np.testing.assert_allclose(blended, 5 / 49)

    def test_feature_dataset_has_one_row_per_number(self) -> None:
        dataset = build_feature_dataset(dated_random_draws(80), min_history=20)
        self.assertEqual(dataset.x.shape[1], 49)
        self.assertTrue(np.all(dataset.y.sum(axis=1) == 5))

    def test_bayesian_nested_backtest_and_abstention(self) -> None:
        draws = dated_random_draws(180)
        results = nested_ml_backtest(
            draws,
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            models=("bayesian",),
        )
        self.assertEqual(results[0].test_draws, 100)
        prediction = predict_next_draw(draws, results, date(2021, 1, 1))
        if not results[0].qualified:
            self.assertEqual(prediction["status"], "abstention")
            self.assertEqual(prediction["game"], "loto")
            self.assertEqual(prediction["training_last_date"], draws[-1].draw_date)
            forced = predict_next_draw(
                draws, results, date(2021, 1, 1), force=True
            )
            self.assertFalse(forced["validation"]["qualified"])
            self.assertIn("mean_brier_delta", forced["validation"])
            self.assertGreaterEqual(forced["probability_spread"], 0.0)

    def test_rolling_bayesian_uses_only_past_window_frequencies(self) -> None:
        draws = dated_random_draws(80)
        dataset = build_feature_dataset(draws, min_history=20)
        parameters = {
            "window": 10,
            "prior_strength": 20.0,
            "uniform_blend": 1.0,
        }
        predicted = _fit_predict(
            "rolling_bayesian",
            parameters,
            dataset.x[:40],
            dataset.y[:40],
            dataset.x[40:41],
            seed=0,
        )
        self.assertAlmostEqual(float(predicted[0].sum()), 5.0)
        self.assertGreater(float(np.ptp(predicted[0])), 0.0)

    def test_rolling_bayesian_participates_in_nested_selection(self) -> None:
        draws = dated_random_draws(180)
        results = nested_ml_backtest(
            draws,
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            models=("rolling_bayesian",),
        )
        self.assertEqual(results[0].model, "rolling_bayesian")
        self.assertEqual(results[0].test_draws, 100)
        self.assertIn(results[0].final_parameters["window"], (10, 50, 200))

    def test_backtest_can_isolate_the_target_game(self) -> None:
        draws = dated_random_draws(240)
        mixed = [
            Draw(
                draw.main,
                draw.chance,
                draw.draw_date,
                "loto" if index % 2 == 0 else "super_loto",
            )
            for index, draw in enumerate(draws)
        ]
        results = nested_ml_backtest(
            mixed,
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            game="super_loto",
            models=("bayesian",),
        )
        self.assertEqual(results[0].test_draws, 40)

    def test_prediction_rejects_a_date_inside_training_history(self) -> None:
        draws = dated_random_draws(180)
        results = nested_ml_backtest(
            draws,
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            models=("bayesian",),
        )
        with self.assertRaisesRegex(ValueError, "posterieure"):
            predict_next_draw(draws, results, draws[-1].draw_date, force=True)

    def test_next_features_ignore_draws_on_or_after_target(self) -> None:
        draws = dated_random_draws(80)
        target = draws[40].draw_date
        expected = next_feature_matrix(draws[:40], target)
        actual = next_feature_matrix(draws, target)
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
