from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from math import exp, expm1
from statistics import median

from .domain import Draw
from .participation import forecast_participation
from .probability import rank_probabilities, total_outcomes

TICKET_PRICES = {"loto": 2.20, "super_loto": 3.0, "grand_loto": 5.0}


@dataclass(frozen=True, slots=True)
class ValueReport:
    game: str
    reference_draws: int
    jackpot: float
    expected_co_winners: float
    expected_co_winners_source: str
    jackpot_share_factor: float
    estimated_tickets: float | None
    participation_model: str | None
    participation_log_rmse: float | None
    lower_rank_ev: float
    code_ev: float
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


def _median_code_pool(draws: list[Draw]) -> float:
    pools = [
        draw.code_winners * draw.code_payout
        for draw in draws
        if draw.code_winners is not None
        and draw.code_payout is not None
        and draw.code_winners > 0
        and draw.code_payout > 0
    ]
    return float(median(pools)) if pools else 0.0


def _poisson_share_factor(expected_other_winners: float) -> float:
    if expected_other_winners == 0:
        return 1.0
    return -expm1(-expected_other_winners) / expected_other_winners


def estimate_value(
    draws: list[Draw],
    jackpot: float,
    game: str = "loto",
    expected_co_winners: float | None = 0.0,
    target_date: date | None = None,
    popularity_factor: float = 1.0,
    participation_min_train: int = 500,
    participation_folds: int = 3,
    bootstrap_simulations: int = 2_000,
    seed: int = 0,
) -> ValueReport:
    if game not in TICKET_PRICES:
        raise ValueError(f"Jeu inconnu: {game}")
    if jackpot < 0 or (expected_co_winners is not None and expected_co_winners < 0):
        raise ValueError("Le jackpot et les co-gagnants ne peuvent pas etre negatifs")
    if popularity_factor < 0:
        raise ValueError("Le facteur de popularite ne peut pas etre negatif")
    if bootstrap_simulations < 100:
        raise ValueError("Il faut au moins 100 simulations bootstrap")
    reference = _current_prize_draws(draws, game)
    if len(reference) < 5:
        raise ValueError(f"Pas assez de tirages recents avec 9 rangs pour {game}")
    probabilities = {item.rank: item.probability for item in rank_probabilities()}
    medians = _rank_medians(reference)
    lower_ev = sum(probabilities[rank] * medians.get(rank, 0.0) for rank in range(2, 10))
    code_pool = _median_code_pool(reference)
    participation = None
    if expected_co_winners is None:
        dated = [draw.draw_date for draw in draws if draw.draw_date is not None]
        forecast_date = target_date or max(dated) + timedelta(days=2)
        participation = forecast_participation(
            draws,
            jackpot,
            forecast_date,
            game,
            participation_min_train,
            participation_folds,
            bootstrap_simulations,
            seed,
        )
        expected_co_winners = (
            participation.estimated_tickets / total_outcomes() * popularity_factor
        )
        share_factor = _poisson_share_factor(expected_co_winners)
        code_ev = code_pool / participation.estimated_tickets
        co_winner_source = "participation_model"
    else:
        share_factor = 1 / (1 + expected_co_winners)
        code_ev = 0.0
        co_winner_source = "manual"
    jackpot_ev = probabilities[1] * jackpot * share_factor
    estimated_ev = lower_ev + code_ev + jackpot_ev
    price = TICKET_PRICES[game]

    rng = random.Random(seed)
    bootstrap_values = []
    for _ in range(bootstrap_simulations):
        sample = [rng.choice(reference) for _ in reference]
        sample_medians = _rank_medians(sample)
        sample_lower = sum(
            probabilities[rank] * sample_medians.get(rank, 0.0) for rank in range(2, 10)
        )
        if participation is None:
            sample_share = share_factor
            sample_code_ev = code_ev
        else:
            sample_tickets = participation.estimated_tickets * exp(
                rng.gauss(0, participation.backtest_log_rmse)
            )
            sample_lambda = sample_tickets / total_outcomes() * popularity_factor
            sample_share = _poisson_share_factor(sample_lambda)
            sample_code_pool = _median_code_pool(sample)
            sample_code_ev = sample_code_pool / sample_tickets
        bootstrap_values.append(
            sample_lower + sample_code_ev + probabilities[1] * jackpot * sample_share
        )
    bootstrap_values.sort()
    ci_low = bootstrap_values[round(0.025 * (len(bootstrap_values) - 1))]
    ci_high = bootstrap_values[round(0.975 * (len(bootstrap_values) - 1))]
    fair_jackpot = max(0.0, (price - lower_ev - code_ev) / probabilities[1] / share_factor)
    return ValueReport(
        game=game,
        reference_draws=len(reference),
        jackpot=jackpot,
        expected_co_winners=expected_co_winners,
        expected_co_winners_source=co_winner_source,
        jackpot_share_factor=share_factor,
        estimated_tickets=(participation.estimated_tickets if participation else None),
        participation_model=(participation.model if participation else None),
        participation_log_rmse=(participation.backtest_log_rmse if participation else None),
        lower_rank_ev=lower_ev,
        code_ev=code_ev,
        estimated_ev=estimated_ev,
        estimated_roi=estimated_ev / price,
        ev_ci_low=ci_low,
        ev_ci_high=ci_high,
        ticket_price=price,
        fair_jackpot=fair_jackpot,
        decision="eligible" if ci_low > price else "no_bet",
        limitations=(
            "Le rang 9 fournit un proxy du nombre de grilles, pas le volume FDJ certifie.",
            "Le partage Poisson suppose une combinaison aussi populaire que la moyenne; "
            "ajuster popularity_factor pour un scenario anti-foule.",
            "Le calcul des codes suppose que leur pool historique se repartit sur les grilles.",
            "Les rapports futurs et le nombre reel de co-gagnants restent inconnus.",
            "Le jackpot d'equilibre fige la participation au niveau du jackpot annonce.",
            "Un point estime positif ne suffit pas: la decision utilise la borne basse a 95%.",
        ),
    )
