import random
import unittest
from datetime import date, timedelta

from loto_lab.domain import Draw, PrizeResult
from loto_lab.participation import (
    forecast_participation,
    participation_backtest,
    participation_observations,
)
from loto_lab.probability import rank_probabilities


def participation_draws(count: int = 180) -> list[Draw]:
    rng = random.Random(2)
    start = date(2020, 1, 1)
    rank_9_probability = {item.rank: item.probability for item in rank_probabilities()}[9]
    draws = []
    for index in range(count):
        jackpot = 2_000_000 + (index % 10) * 1_000_000
        tickets = 2_000_000 + jackpot * 0.2 + (index % 7) * 100_000
        winners = round(tickets * rank_9_probability)
        prizes = tuple(
            PrizeResult(rank, winners if rank == 9 else 0, jackpot if rank == 1 else 10)
            for rank in range(1, 10)
        )
        draws.append(
            Draw(
                tuple(rng.sample(range(1, 50), 5)),
                rng.randint(1, 10),
                start + timedelta(days=index),
                "loto",
                prizes,
            )
        )
    return draws


class ParticipationTests(unittest.TestCase):
    def test_rank_nine_winners_infer_ticket_volume(self) -> None:
        observations = participation_observations(participation_draws())
        self.assertEqual(len(observations), 180)
        self.assertGreater(observations[0].estimated_tickets, 2_000_000)

    def test_temporal_backtest_and_forecast(self) -> None:
        draws = participation_draws()
        results = participation_backtest(draws, min_train=100, folds=2, simulations=100)
        self.assertEqual({result.model for result in results}, {"ridge", "gradient_boosting"})
        forecast = forecast_participation(
            draws,
            5_000_000,
            date(2021, 1, 1),
            min_train=100,
            folds=2,
            simulations=100,
        )
        self.assertGreater(forecast.estimated_tickets, 0)


if __name__ == "__main__":
    unittest.main()
