import random
import unittest
from datetime import date, timedelta

import numpy as np

from loto_lab.domain import Draw
from loto_lab.ml import (
    blend_with_uniform,
    build_feature_dataset,
    nested_ml_backtest,
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


if __name__ == "__main__":
    unittest.main()
