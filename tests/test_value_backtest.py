import unittest

from test_participation import participation_draws

from loto_lab.domain import Draw
from loto_lab.value_backtest import backtest_value


class ValueBacktestTests(unittest.TestCase):
    def test_walk_forward_value_backtest(self) -> None:
        draws = [
            Draw(
                draw.main,
                draw.chance,
                draw.draw_date,
                draw.game,
                draw.prizes,
                10,
                20_000,
            )
            for draw in participation_draws(300)
        ]
        result = backtest_value(
            draws,
            min_train=200,
            folds=2,
            simulations=100,
            refit_interval=25,
        )
        self.assertEqual(result.observations, 100)
        self.assertEqual(result.refits, 4)
        self.assertEqual(result.refit_interval_dates, 25)
        self.assertEqual(len(result.periods), 4)
        self.assertTrue(
            all(period.training_last_date < period.first_test_date for period in result.periods)
        )
        self.assertGreaterEqual(result.prediction_interval_coverage, 0)
        self.assertLessEqual(result.prediction_interval_coverage, 1)
        self.assertLessEqual(result.coverage_ci_low, result.prediction_interval_coverage)
        self.assertGreaterEqual(result.coverage_ci_high, result.prediction_interval_coverage)
        self.assertGreater(result.mean_prediction_interval_width, 0)
        self.assertEqual(result.temporal_block_size, 12)
        self.assertIn("block", result.inference_method)
        self.assertIsInstance(result.coverage_compatible_with_target, bool)
        self.assertEqual(
            result.value_model_qualified,
            result.qualified_against_naive and result.coverage_compatible_with_target,
        )
        self.assertGreater(result.naive_mae, 0)
        self.assertLessEqual(result.mae_delta_ci_low, result.mae_delta_ci_high)

    def test_block_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "block_size"):
            backtest_value([], min_train=200, folds=2, simulations=100, block_size=0)

    def test_refit_interval_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "refit_interval"):
            backtest_value([], min_train=200, folds=2, simulations=100, refit_interval=0)


if __name__ == "__main__":
    unittest.main()
