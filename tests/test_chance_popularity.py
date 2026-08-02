import random
import unittest
from datetime import date, timedelta

from loto_lab.chance_popularity import chance_observations, chance_popularity_backtest
from loto_lab.domain import Draw, PrizeResult


def chance_draws(count: int = 240) -> list[Draw]:
    rng = random.Random(12)
    start = date(2020, 1, 1)
    draws = []
    for index in range(count):
        chance = rng.randint(1, 10)
        correct = 25 if chance == 5 else 5
        draws.append(
            Draw(
                tuple(rng.sample(range(1, 50), 5)),
                chance,
                start + timedelta(days=index),
                prizes=(
                    PrizeResult(1, correct, 2_000_000.0),
                    PrizeResult(2, 100 - correct, 100_000.0),
                ),
            )
        )
    return draws


class ChancePopularityTests(unittest.TestCase):
    def test_observations_use_rank_one_and_two(self) -> None:
        observations = chance_observations(chance_draws(3))
        self.assertEqual(len(observations), 3)
        self.assertEqual(
            observations[0].main_combination_winners,
            observations[0].correct_chance_winners + 100 - observations[0].correct_chance_winners,
        )

    def test_temporal_backtest_detects_synthetic_preference(self) -> None:
        result = chance_popularity_backtest(
            chance_draws(), min_train=100, outer_folds=2, simulations=100, seed=12
        )
        self.assertEqual(result.test_observations, 140)
        self.assertLess(result.deviance_delta, 0)
        self.assertLess(result.delta_ci_high, 0)
        self.assertTrue(result.qualified)
        self.assertGreater(result.chance_factors[4], 1)


if __name__ == "__main__":
    unittest.main()
