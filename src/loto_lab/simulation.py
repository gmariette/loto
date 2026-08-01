from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .domain import DEFAULT_RULES, Draw, LotteryRules, winning_rank
from .strategy import generate_tickets


@dataclass(frozen=True, slots=True)
class SimulationResult:
    runs: int
    draws_per_run: int
    tickets_per_draw: int
    cost_per_run: float
    mean_payout: float
    mean_net: float
    median_net: float
    positive_run_rate: float
    p05_net: float
    p95_net: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def load_payouts(path: str | Path) -> dict[int, float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    payouts = {int(rank): float(amount) for rank, amount in raw["payouts"].items()}
    if set(payouts) != set(range(1, 10)):
        raise ValueError("Le fichier doit definir les rangs 1 a 9")
    if any(amount < 0 for amount in payouts.values()):
        raise ValueError("Les gains ne peuvent pas etre negatifs")
    return payouts


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def simulate_bankroll(
    payouts: dict[int, float],
    tickets_per_draw: int,
    draws_per_run: int,
    runs: int,
    seed: int = 0,
    rules: LotteryRules = DEFAULT_RULES,
) -> SimulationResult:
    if min(tickets_per_draw, draws_per_run, runs) < 1:
        raise ValueError("tickets, tirages et runs doivent etre positifs")
    rng = random.Random(seed)
    tickets = generate_tickets(tickets_per_draw, seed=seed, anti_crowd=False, rules=rules)
    cost = tickets_per_draw * draws_per_run * rules.ticket_price
    net_results: list[float] = []
    payouts_by_run: list[float] = []
    population = range(1, rules.main_pool + 1)
    for _ in range(runs):
        payout = 0.0
        for _ in range(draws_per_run):
            draw = Draw(
                tuple(rng.sample(population, rules.main_drawn)),
                rng.randint(1, rules.chance_pool),
            )
            for ticket in tickets:
                rank = winning_rank(ticket, draw)
                if rank is not None:
                    payout += payouts[rank]
        payouts_by_run.append(payout)
        net_results.append(payout - cost)
    return SimulationResult(
        runs=runs,
        draws_per_run=draws_per_run,
        tickets_per_draw=tickets_per_draw,
        cost_per_run=cost,
        mean_payout=sum(payouts_by_run) / runs,
        mean_net=sum(net_results) / runs,
        median_net=_quantile(net_results, 0.5),
        positive_run_rate=sum(value > 0 for value in net_results) / runs,
        p05_net=_quantile(net_results, 0.05),
        p95_net=_quantile(net_results, 0.95),
    )
