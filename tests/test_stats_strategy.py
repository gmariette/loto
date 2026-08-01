import random
import unittest

from loto_lab.domain import Draw
from loto_lab.stats import lag_overlap, main_uniformity, pair_frequency_outliers
from loto_lab.strategy import anti_crowd_score, generate_tickets


class StatsAndStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = random.Random(7)
        self.draws = [
            Draw(tuple(rng.sample(range(1, 50), 5)), rng.randint(1, 10)) for _ in range(60)
        ]

    def test_uniformity_is_reproducible(self) -> None:
        first = main_uniformity(self.draws, simulations=25, seed=5)
        second = main_uniformity(self.draws, simulations=25, seed=5)
        self.assertEqual(first.monte_carlo_p_value, second.monte_carlo_p_value)
        self.assertEqual(sum(first.counts.values()), 300)

    def test_overlap_and_pairs(self) -> None:
        overlap = lag_overlap(self.draws)
        self.assertEqual(overlap["transitions"], 59)
        self.assertEqual(len(pair_frequency_outliers(self.draws, limit=5)), 5)

    def test_generator_is_deterministic_distinct_and_balanced(self) -> None:
        first = generate_tickets(12, seed=12)
        second = generate_tickets(12, seed=12)
        self.assertEqual(first, second)
        self.assertEqual(len({ticket.main for ticket in first}), 12)
        counts = {
            chance: sum(ticket.chance == chance for ticket in first)
            for chance in range(1, 11)
        }
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_anti_crowd_score_prefers_numbers_above_birthdays(self) -> None:
        self.assertGreater(
            anti_crowd_score((32, 36, 41, 45, 49)),
            anti_crowd_score((1, 2, 3, 4, 5)),
        )


if __name__ == "__main__":
    unittest.main()
