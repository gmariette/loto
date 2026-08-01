import random
import unittest

from loto_lab import DEFAULT_RULES
from loto_lab.domain import Draw
from loto_lab.models import (
    DecayedFrequencyPredictor,
    SmoothedFrequencyPredictor,
    walk_forward_backtest,
)


def random_draws(count: int, seed: int = 42) -> list[Draw]:
    rng = random.Random(seed)
    return [
        Draw(tuple(rng.sample(range(1, 50), 5)), rng.randint(1, 10)) for _ in range(count)
    ]


class ModelTests(unittest.TestCase):
    def test_predictors_produce_valid_marginals(self) -> None:
        draws = random_draws(100)
        for predictor in (SmoothedFrequencyPredictor(), DecayedFrequencyPredictor()):
            probabilities = predictor.predict(draws, rules=DEFAULT_RULES)
            self.assertEqual(len(probabilities), 49)
            self.assertAlmostEqual(sum(probabilities), 5)
            self.assertTrue(all(0 < value < 1 for value in probabilities))

    def test_walk_forward_backtest(self) -> None:
        result = walk_forward_backtest(
            random_draws(80), lambda: SmoothedFrequencyPredictor(), min_train=30
        )
        self.assertEqual(result.test_draws, 50)
        self.assertGreater(result.mean_brier, 0)
        self.assertGreaterEqual(result.p_value_two_sided, 0)
        self.assertLessEqual(result.p_value_two_sided, 1)


if __name__ == "__main__":
    unittest.main()
