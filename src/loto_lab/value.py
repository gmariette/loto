from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from statistics import median

from .domain import Draw
from .probability import rank_probabilities

TICKET_PRICES = {"loto": 2.20, "super_loto": 3.0, "grand_loto": 5.0}


@dataclass(frozen=True, slots=True)
class ValueReport:
    game: str
    reference_draws: int
    jackpot: float
    expected_co_winners: float
    lower_rank_ev: float
    estimated_ev: float
    estimated_roi: float
    ev_ci_low: float
    ev_ci_high: float
    ticket_price: float
    fair_jackpot: float
    decision: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _current_prize_draws(draws: list[Draw], game: str) -> list[Draw]:
    selected = []
    for draw in draws:
        ranks = {prize.rank for prize in draw.prizes if prize.payout is not None}
        if draw.game == game and ranks.issuperset(range(1, 10)):
            selected.append(draw)
    return selected


def _rank_medians(draws: list[Draw]) -> dict[int, float]:
    values: dict[int, list[float]] = {rank: [] for rank in range(1, 10)}
    for draw in draws:
        for prize in draw.prizes:
            if prize.payout is not None and prize.payout > 0 and prize.rank in values:
                values[prize.rank].append(prize.payout)
    return {rank: median(payouts) for rank, payouts in values.items() if payouts}


def estimate_value(
    draws: list[Draw],
    jackpot: float,
    game: str = "loto",
    expected_co_winners: float = 0.0,
    bootstrap_simulations: int = 2_000,
    seed: int = 0,
) -> ValueReport:
    if game not in TICKET_PRICES:
        raise ValueError(f"Jeu inconnu: {game}")
    if jackpot < 0 or expected_co_winners < 0:
        raise ValueError("Le jackpot et les co-gagnants ne peuvent pas etre negatifs")
    if bootstrap_simulations < 100:
        raise ValueError("Il faut au moins 100 simulations bootstrap")
    reference = _current_prize_draws(draws, game)
    if len(reference) < 5:
        raise ValueError(f"Pas assez de tirages recents avec 9 rangs pour {game}")
    probabilities = {item.rank: item.probability for item in rank_probabilities()}
    medians = _rank_medians(reference)
    lower_ev = sum(probabilities[rank] * medians.get(rank, 0.0) for rank in range(2, 10))
    jackpot_ev = probabilities[1] * jackpot / (1 + expected_co_winners)
    estimated_ev = lower_ev + jackpot_ev
    price = TICKET_PRICES[game]

    rng = random.Random(seed)
    bootstrap_values = []
    for _ in range(bootstrap_simulations):
        sample = [rng.choice(reference) for _ in reference]
        sample_medians = _rank_medians(sample)
        sample_lower = sum(
            probabilities[rank] * sample_medians.get(rank, 0.0) for rank in range(2, 10)
        )
        bootstrap_values.append(sample_lower + jackpot_ev)
    bootstrap_values.sort()
    ci_low = bootstrap_values[round(0.025 * (len(bootstrap_values) - 1))]
    ci_high = bootstrap_values[round(0.975 * (len(bootstrap_values) - 1))]
    fair_jackpot = max(
        0.0, (price - lower_ev) * (1 + expected_co_winners) / probabilities[1]
    )
    return ValueReport(
        game=game,
        reference_draws=len(reference),
        jackpot=jackpot,
        expected_co_winners=expected_co_winners,
        lower_rank_ev=lower_ev,
        estimated_ev=estimated_ev,
        estimated_roi=estimated_ev / price,
        ev_ci_low=ci_low,
        ev_ci_high=ci_high,
        ticket_price=price,
        fair_jackpot=fair_jackpot,
        decision="eligible" if ci_low > price else "no_bet",
        limitations=(
            "Les codes gagnants ne sont pas modelises faute du nombre total de prises de jeu.",
            "Les rapports futurs et le nombre reel de co-gagnants sont inconnus.",
            "Un point estime positif ne suffit pas: la decision utilise la borne basse a 95%.",
        ),
    )
