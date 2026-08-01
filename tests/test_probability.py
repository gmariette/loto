import unittest

from loto_lab.domain import Draw, Ticket, winning_rank
from loto_lab.probability import (
    expected_budget,
    probability_of_any_prize,
    rank_probabilities,
    total_outcomes,
)


class ProbabilityTests(unittest.TestCase):
    def test_exact_outcomes_and_jackpot(self) -> None:
        self.assertEqual(total_outcomes(), 19_068_840)
        jackpot = rank_probabilities()[0]
        self.assertEqual(jackpot.rank, 1)
        self.assertEqual(jackpot.outcomes, 1)
        self.assertEqual(jackpot.one_in, 19_068_840)

    def test_any_prize_probability_matches_official_odds(self) -> None:
        probability = probability_of_any_prize()
        self.assertAlmostEqual(1 / probability, 5.9852484626)

    def test_rank_probabilities_are_disjoint(self) -> None:
        ranks = rank_probabilities()
        self.assertEqual({rank.rank for rank in ranks}, set(range(1, 10)))
        self.assertAlmostEqual(sum(rank.probability for rank in ranks), probability_of_any_prize())

    def test_expected_budget_uses_current_global_return_rate(self) -> None:
        result = expected_budget(220)
        self.assertAlmostEqual(result["expected_return"], 119.57)
        self.assertAlmostEqual(result["expected_loss"], 100.43)

    def test_winning_ranks(self) -> None:
        draw = Draw((1, 2, 3, 4, 5), 6)
        self.assertEqual(winning_rank(Ticket((1, 2, 3, 4, 5), 6), draw), 1)
        self.assertEqual(winning_rank(Ticket((1, 2, 3, 4, 5), 7), draw), 2)
        self.assertEqual(winning_rank(Ticket((1, 2, 10, 11, 12), 7), draw), 8)
        self.assertEqual(winning_rank(Ticket((10, 11, 12, 13, 14), 6), draw), 9)
        self.assertIsNone(winning_rank(Ticket((10, 11, 12, 13, 14), 7), draw))


if __name__ == "__main__":
    unittest.main()
