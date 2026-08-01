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
        result = backtest_value(draws, min_train=200, folds=2, simulations=100)
        self.assertEqual(result.observations, 100)
        self.assertGreaterEqual(result.prediction_interval_coverage, 0)
        self.assertLessEqual(result.prediction_interval_coverage, 1)
        self.assertLessEqual(result.coverage_ci_low, result.prediction_interval_coverage)
        self.assertGreaterEqual(result.coverage_ci_high, result.prediction_interval_coverage)
        self.assertGreater(result.mean_prediction_interval_width, 0)
        self.assertEqual(result.temporal_block_size, 12)
        self.assertIn("block", result.inference_method)
        self.assertGreater(result.naive_mae, 0)
        self.assertLessEqual(result.mae_delta_ci_low, result.mae_delta_ci_high)

    def test_block_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "block_size"):
            backtest_value([], min_train=200, folds=2, simulations=100, block_size=0)


if __name__ == "__main__":
    unittest.main()
