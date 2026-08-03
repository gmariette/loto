import random
import unittest
from datetime import date, timedelta

import numpy as np

from loto_lab.chance_ml import (
    _expected_top1_hits,
    argmax_with_ties,
    chance_ml_backtest,
    predict_chance,
    uniform_brier,
    uniform_log_loss,
    uniform_probability,
)
from loto_lab.domain import DEFAULT_RULES, Draw


def alternating_chance_draws(count: int = 300) -> list[Draw]:
    start = date(2020, 1, 1)
    return [
        Draw(
            (1, 2, 3, 4, 5),
            1 if index % 2 == 0 else 2,
            start + timedelta(days=index),
        )
        for index in range(count)
    ]


def uniform_chance_draws(count: int = 600, seed: int = 3) -> list[Draw]:
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


class ChanceReferenceTests(unittest.TestCase):
    def test_uniform_references_match_their_closed_forms(self) -> None:
        pool = DEFAULT_RULES.chance_pool
        self.assertAlmostEqual(uniform_probability(pool), 0.1)
        # Brier moyenne sur les classes: ((1/K - 1)^2 + (K-1)/K^2) / K = (K-1)/K^2.
        self.assertAlmostEqual(uniform_brier(pool), 0.09)
        self.assertAlmostEqual(uniform_log_loss(pool), float(np.log(10)))

    def test_argmax_reports_every_tied_candidate(self) -> None:
        probabilities = np.full(10, 0.1)
        number, tied = argmax_with_ties(probabilities)
        self.assertEqual(number, 1)
        self.assertEqual(list(tied), list(range(1, 11)))

    def test_expected_top1_shares_credit_between_ties(self) -> None:
        probabilities = np.tile(np.array([0.2, 0.2, *([0.075] * 8)]), (2, 1))
        hits, tie_sizes = _expected_top1_hits(probabilities, np.array([0, 5]))
        np.testing.assert_allclose(hits, [0.5, 0.0])
        np.testing.assert_array_equal(tie_sizes, [2, 2])


class ChanceMLTests(unittest.TestCase):
    def test_temporal_transition_model_beats_uniform_on_predictable_sequence(self) -> None:
        draws = alternating_chance_draws()
        result = chance_ml_backtest(
            draws,
            min_train=100,
            outer_folds=2,
            simulations=200,
            block_size=6,
            seed=42,
        )
        self.assertLess(result.mean_brier_delta, 0)
        self.assertLess(result.delta_ci_high, 0)
        self.assertLess(result.mean_log_loss, result.uniform_log_loss)
        self.assertGreater(result.top1_accuracy, result.uniform_top1_accuracy)
        self.assertTrue(result.qualified)
        self.assertGreater(float(result.final_parameters["transition_weight"]), 0)

    def test_qualified_model_publishes_its_own_distribution(self) -> None:
        draws = alternating_chance_draws()
        result = chance_ml_backtest(
            draws, min_train=100, outer_folds=2, simulations=200, seed=7
        )
        target = draws[-1].draw_date + timedelta(days=1)
        first = predict_chance(draws, target, validation=result)
        second = predict_chance(draws, target, validation=result)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "qualified")
        self.assertEqual(first["number"], 1)
        self.assertEqual(first["selection_method"], "temporal_dirichlet_argmax")
        self.assertAlmostEqual(sum(first["probabilities"].values()), 1.0)

    def test_iid_uniform_chance_does_not_qualify(self) -> None:
        result = chance_ml_backtest(
            uniform_chance_draws(),
            min_train=200,
            outer_folds=2,
            simulations=300,
            seed=5,
        )
        self.assertFalse(result.qualified)
        self.assertEqual(result.qualification_basis, "none")
        self.assertIsInstance(result.qualified, bool)

    def test_unqualified_model_abstains_with_ten_uniform_probabilities(self) -> None:
        draws = uniform_chance_draws()
        result = chance_ml_backtest(
            draws, min_train=200, outer_folds=2, simulations=300, seed=5
        )
        self.assertFalse(result.qualified)
        prediction = predict_chance(
            draws, draws[-1].draw_date + timedelta(days=1), validation=result
        )
        self.assertEqual(prediction["status"], "abstention")
        self.assertIsNone(prediction["number"])
        self.assertEqual(prediction["selection_method"], "uniform_abstention")
        self.assertEqual(len(prediction["probabilities"]), 10)
        for probability in prediction["probabilities"].values():
            self.assertEqual(probability, 0.1)
        self.assertAlmostEqual(prediction["probability_sum"], 1.0)
        experimental = prediction["experimental"]
        self.assertIsNotNone(experimental["number"])
        self.assertNotEqual(
            experimental["probabilities"], prediction["probabilities"]
        )

    def test_missing_validation_also_abstains(self) -> None:
        draws = uniform_chance_draws(count=300)
        prediction = predict_chance(draws, draws[-1].draw_date + timedelta(days=1))
        self.assertEqual(prediction["status"], "abstention")
        self.assertIsNone(prediction["number"])
        self.assertIsNone(prediction["validation"])

    def test_result_flags_are_json_safe_booleans(self) -> None:
        import json

        result = chance_ml_backtest(
            uniform_chance_draws(),
            min_train=200,
            outer_folds=2,
            simulations=200,
            seed=1,
        )
        payload = json.loads(json.dumps(result.to_dict(), default=str))
        self.assertIs(payload["qualified"], False)

    def test_probabilities_sum_to_one_for_every_grid_point(self) -> None:
        from loto_lab.chance_ml import CHANCE_PARAMETER_GRID, _prediction_series

        draws = uniform_chance_draws(count=200)
        chances = np.asarray([draw.chance - 1 for draw in draws])
        weekdays = np.asarray([draw.draw_date.weekday() for draw in draws])
        for parameters in CHANCE_PARAMETER_GRID:
            series = _prediction_series(chances, weekdays, parameters)
            np.testing.assert_allclose(series.sum(axis=1), 1.0, atol=1e-12)

    def test_short_history_selection_falls_back_instead_of_crashing(self) -> None:
        from loto_lab.chance_ml import DEFAULT_CHANCE_PARAMETERS, _select_parameters

        dates = np.array([1, 1, 1])
        chances = np.array([0, 1, 2])
        parameters = _select_parameters(dates, chances, {}, np.arange(3))
        self.assertEqual(parameters, DEFAULT_CHANCE_PARAMETERS)


if __name__ == "__main__":
    unittest.main()
