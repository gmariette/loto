import random
import unittest
from datetime import date, timedelta

import numpy as np

from loto_lab.domain import Draw, PrizeResult
from loto_lab.popularity import (
    fit_popularity_predictor,
    optimize_value_aware_ticket,
    popularity_backtest,
    popularity_observations,
)


def popularity_draws(count: int = 240) -> list[Draw]:
    rng = random.Random(44)
    start = date(2020, 1, 1)
    draws = []
    for index in range(count):
        main = tuple(rng.sample(range(1, 50), 5))
        jackpot_winners = 2 if all(number <= 31 for number in main) else 0
        draws.append(
            Draw(
                main,
                rng.randint(1, 10),
                start + timedelta(days=index),
                prizes=(
                    PrizeResult(1, jackpot_winners, 2_000_000.0),
                    PrizeResult(2, 3 if jackpot_winners else 0, 100_000.0),
                    PrizeResult(9, 100_000, 2.2),
                ),
            )
        )
    return draws


class PopularityTests(unittest.TestCase):
    def test_observations_use_rank9_as_ticket_exposure(self) -> None:
        observations = popularity_observations(popularity_draws(3))
        self.assertEqual(len(observations), 3)
        self.assertGreater(observations[0].estimated_tickets, 0)
        self.assertGreater(observations[0].exposure, 0)

    def test_main_combination_target_uses_ranks_one_and_two(self) -> None:
        draws = popularity_draws()
        jackpot = popularity_observations(draws, target="jackpot")
        main = popularity_observations(draws, target="main_combination")
        self.assertAlmostEqual(main[0].exposure, jackpot[0].exposure * 10)
        self.assertEqual(
            main[0].jackpot_winners,
            jackpot[0].jackpot_winners + draws[0].prizes[1].winners,
        )
        result = popularity_backtest(
            draws,
            target="main_combination",
            min_train=100,
            outer_folds=2,
            simulations=100,
            block_size=6,
            seed=12,
        )
        self.assertEqual(result.target, "main_combination")
        self.assertEqual(result.adjusted_p_value, min(1.0, 2 * result.permutation_p_value))
        self.assertTrue(result.qualified)

    def test_temporal_backtest_and_value_aware_constraint(self) -> None:
        draws = popularity_draws()
        result = popularity_backtest(
            draws,
            min_train=100,
            outer_folds=2,
            simulations=100,
            block_size=6,
            seed=12,
        )
        self.assertEqual(result.test_observations, 140)
        self.assertTrue(np.isfinite(result.deviance_delta))
        predictor = fit_popularity_predictor(
            draws, result, bootstrap_models=20, seed=12
        )
        probabilities = np.linspace(0.01, 0.49, 49)
        selection = optimize_value_aware_ticket(
            probabilities,
            predictor,
            max_expected_hit_loss=0.0,
            chance=3,
        )
        self.assertEqual(selection.ticket.main, (45, 46, 47, 48, 49))
        self.assertEqual(selection.ticket.chance, 3)
        self.assertAlmostEqual(selection.expected_hit_loss, 0.0)
        self.assertTrue(np.isfinite(selection.conservative_popularity_multiplier))
        self.assertEqual(selection.bootstrap_models, 20)
        repeated = fit_popularity_predictor(
            draws, result, bootstrap_models=20, seed=12
        )
        np.testing.assert_allclose(
            predictor.bootstrap_coefficients, repeated.bootstrap_coefficients
        )
        self.assertEqual(selection.combinations_evaluated, 1_906_884)

    def test_optimizer_rejects_negative_loss_budget(self) -> None:
        draws = popularity_draws()
        result = popularity_backtest(
            draws, min_train=100, simulations=100, outer_folds=2
        )
        predictor = fit_popularity_predictor(
            draws, result, bootstrap_models=20, seed=12
        )
        with self.assertRaisesRegex(ValueError, "positifs"):
            optimize_value_aware_ticket(
                np.full(49, 5 / 49), predictor, max_expected_hit_loss=-0.1
            )


if __name__ == "__main__":
    unittest.main()
