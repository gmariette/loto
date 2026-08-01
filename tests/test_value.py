import unittest
from datetime import date, timedelta

from loto_lab.domain import Draw, PrizeResult
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


if __name__ == "__main__":
    unittest.main()
