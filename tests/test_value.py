import unittest
from datetime import date, timedelta

from test_participation import participation_draws

from loto_lab.domain import Draw, PrizeResult
from loto_lab.probability import rank_probabilities
from loto_lab.value import estimate_value


class ValueTests(unittest.TestCase):
    def test_value_report_uses_nine_rank_draws(self) -> None:
        prizes = tuple(PrizeResult(rank, 1, 100 / rank) for rank in range(1, 10))
        draws = [
            Draw((1, 2, 3, 4, 5), 1, date(2020, 1, 1) + timedelta(days=i), "loto", prizes)
            for i in range(10)
        ]
        result = estimate_value(draws, jackpot=2_000_000, bootstrap_simulations=100)
        self.assertEqual(result.reference_draws, 10)
        self.assertGreater(result.estimated_ev, 0)
        self.assertGreater(result.fair_jackpot, 0)
        self.assertAlmostEqual(
            result.expected_return_rate, result.estimated_ev / result.ticket_price
        )
        self.assertAlmostEqual(result.estimated_roi, result.expected_return_rate - 1)
        self.assertGreater(result.naive_ev, 0)

    def test_automatic_participation_models_sharing_and_codes(self) -> None:
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
            for draw in participation_draws()
        ]
        result = estimate_value(
            draws,
            jackpot=5_000_000,
            expected_co_winners=None,
            target_date=date(2021, 1, 1),
            participation_min_train=100,
            participation_folds=2,
            bootstrap_simulations=100,
        )
        self.assertEqual(result.expected_co_winners_source, "participation_model")
        self.assertEqual(
            result.participation_uncertainty_method,
            "temporal_empirical_residuals",
        )
        self.assertGreater(result.participation_residual_observations or 0, 0)
        self.assertGreater(result.estimated_tickets or 0, 0)
        self.assertGreater(result.code_ev, 0)
        self.assertLess(result.jackpot_share_factor, 1)
        self.assertGreaterEqual(result.conservative_jackpot or 0, result.fair_jackpot)
        self.assertEqual(
            result.uncertainty_method,
            "temporally_selected_payout_and_empirical_residual_bootstrap",
        )
        self.assertGreater(result.predictive_payout_window, 0)
        self.assertIn(result.predictive_tail_probability, (0.01, 0.025, 0.05))
        self.assertGreaterEqual(result.payout_validation_coverage, 0)
        self.assertEqual(result.prediction_interval_target, 0.95)

    def test_lower_rank_ev_averages_complete_draw_values(self) -> None:
        draws = []
        for index, payout in enumerate((1.0, 101.0) * 3):
            prizes = tuple(
                PrizeResult(rank, 1, 1_000_000 if rank == 1 else payout)
                for rank in range(1, 10)
            )
            draws.append(
                Draw(
                    (1, 2, 3, 4, 5),
                    1,
                    date(2020, 1, 1) + timedelta(days=index),
                    "loto",
                    prizes,
                )
            )
        result = estimate_value(draws, jackpot=2_000_000, bootstrap_simulations=100)
        lower_probability = sum(item.probability for item in rank_probabilities() if item.rank > 1)
        self.assertAlmostEqual(result.lower_rank_ev, 51 * lower_probability)

    def test_payout_bootstrap_is_invariant_to_input_order(self) -> None:
        draws = participation_draws(30)
        ordered = estimate_value(draws, jackpot=2_000_000, bootstrap_simulations=100, seed=7)
        reversed_report = estimate_value(
            list(reversed(draws)), jackpot=2_000_000, bootstrap_simulations=100, seed=7
        )
        self.assertEqual(ordered.predictive_payout_window, reversed_report.predictive_payout_window)
        self.assertAlmostEqual(ordered.ev_ci_low, reversed_report.ev_ci_low)
        self.assertAlmostEqual(ordered.ev_ci_high, reversed_report.ev_ci_high)

    def test_value_target_date_discards_future_payouts(self) -> None:
        draws = participation_draws(180)
        cutoff = draws[100].draw_date
        result = estimate_value(
            draws,
            jackpot=2_000_000,
            target_date=cutoff,
            bootstrap_simulations=100,
        )
        self.assertEqual(result.reference_draws, 100)
        self.assertLess(result.training_last_date, cutoff)


if __name__ == "__main__":
    unittest.main()
