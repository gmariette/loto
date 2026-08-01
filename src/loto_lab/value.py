from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from math import expm1
from statistics import fmean

from .domain import Draw
from .participation import ParticipationForecaster, fit_participation_forecaster
from .probability import rank_probabilities, total_outcomes

TICKET_PRICES = {"loto": 2.20, "super_loto": 3.0, "grand_loto": 5.0}


@dataclass(frozen=True, slots=True)
class ValueReport:
    game: str
    target_date: date | None
    training_last_date: date
    reference_draws: int
    jackpot: float
    expected_co_winners: float
    expected_co_winners_source: str
    jackpot_share_factor: float
    estimated_tickets: float | None
    median_estimated_tickets: float | None
    participation_smearing_factor: float | None
    participation_model: str | None
    participation_log_rmse: float | None
    participation_uncertainty_method: str | None
    participation_residual_observations: int | None
    participation_jackpot_max: float | None
    participation_extrapolated: bool
    lower_rank_ev: float
    lower_rank_ev_method: str
    predictive_payout_window: int
    predictive_tail_probability: float
    payout_validation_coverage: float
    prediction_interval_target: float
    code_ev: float
    estimated_ev: float
    naive_ev: float
    expected_return_rate: float
    estimated_roi: float
    ev_ci_low: float
    ev_ci_high: float
    ticket_price: float
    fair_jackpot: float
    fair_jackpot_extrapolated: bool
    conservative_jackpot: float | None
    conservative_jackpot_extrapolated: bool
    uncertainty_method: str
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
    return sorted(selected, key=lambda draw: draw.draw_date or date.min)


def _lower_rank_ev(draw: Draw, probabilities: dict[int, float]) -> float:
    payouts = {
        prize.rank: prize.payout
        for prize in draw.prizes
        if prize.payout is not None and prize.payout >= 0
    }
    return sum(probabilities[rank] * payouts.get(rank, 0.0) for rank in range(2, 10))


def _mean_code_pool(draws: list[Draw]) -> float:
    pools = [
        draw.code_winners * draw.code_payout
        for draw in draws
        if draw.code_winners is not None
        and draw.code_payout is not None
        and draw.code_winners > 0
        and draw.code_payout > 0
    ]
    return float(fmean(pools)) if pools else 0.0


def _select_predictive_reference(
    draws: list[Draw], probabilities: dict[int, float]
) -> tuple[list[Draw], int, float, float]:
    if len(draws) < 20:
        return draws, len(draws), 1.0, 0.025
    cutoff = max(10, int(len(draws) * 0.8))
    base = draws[:cutoff]
    validation_values = [_lower_rank_ev(draw, probabilities) for draw in draws[cutoff:]]
    candidates = sorted({min(window, len(base)) for window in (50, 100, 250, len(base))})
    best: tuple[tuple[float, float], int, float, float] | None = None
    for window in candidates:
        values = [_lower_rank_ev(draw, probabilities) for draw in base[-window:]]
        for tail_probability in (0.01, 0.025, 0.05):
            low = _quantile(values, tail_probability)
            high = _quantile(values, 1 - tail_probability)
            coverage = sum(low <= value <= high for value in validation_values) / len(
                validation_values
            )
            score = (abs(coverage - 0.95), high - low)
            if best is None or score < best[0]:
                best = (score, window, coverage, tail_probability)
    assert best is not None
    return draws[-best[1] :], best[1], best[2], best[3]


def _poisson_share_factor(expected_other_winners: float) -> float:
    if expected_other_winners == 0:
        return 1.0
    return -expm1(-expected_other_winners) / expected_other_winners


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[round(probability * (len(ordered) - 1))]


def _solve_threshold(
    value_at: Callable[[float], float], ticket_price: float, initial_high: float
) -> float | None:
    low = 1.0
    high = max(initial_high, 1_000_000.0)
    while value_at(high) < ticket_price and high < 1_000_000_000:
        low = high
        high *= 2
    if value_at(high) < ticket_price:
        return None
    for _ in range(60):
        middle = (low + high) / 2
        if value_at(middle) >= ticket_price:
            high = middle
        else:
            low = middle
    return high


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
    historical = [
        draw
        for draw in draws
        if draw.draw_date is not None
        and (target_date is None or draw.draw_date < target_date)
    ]
    if not historical:
        raise ValueError("Aucun tirage strictement anterieur a la date cible")
    training_last_date = max(
        draw.draw_date for draw in historical if draw.draw_date is not None
    )
    report_target_date = target_date or training_last_date + timedelta(days=2)
    reference = _current_prize_draws(historical, game)
    if len(reference) < 5:
        raise ValueError(f"Pas assez de tirages recents avec 9 rangs pour {game}")
    probabilities = {item.rank: item.probability for item in rank_probabilities()}
    lower_values = [_lower_rank_ev(draw, probabilities) for draw in reference]
    lower_ev = float(fmean(lower_values))
    code_pool = _mean_code_pool(reference)
    predictive_reference, payout_window, payout_validation_coverage, tail_probability = (
        _select_predictive_reference(reference, probabilities)
    )
    predictive_lower_values = [
        _lower_rank_ev(draw, probabilities) for draw in predictive_reference
    ]
    participation = None
    forecaster: ParticipationForecaster | None = None
    forecast_date = report_target_date
    if expected_co_winners is None:
        forecaster = fit_participation_forecaster(
            historical,
            participation_min_train,
            participation_folds,
            bootstrap_simulations,
            seed,
        )
        participation = forecaster.forecast(jackpot, forecast_date, game)
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
    naive_ev = lower_ev + probabilities[1] * jackpot
    price = TICKET_PRICES[game]
    expected_return_rate = estimated_ev / price

    rng = random.Random(seed)
    sample_indices = [
        rng.randrange(len(predictive_reference)) for _ in range(bootstrap_simulations)
    ]
    sample_lower_values = [predictive_lower_values[index] for index in sample_indices]
    sample_code_pools = [
        (
            predictive_reference[index].code_winners * predictive_reference[index].code_payout
            if predictive_reference[index].code_winners is not None
            and predictive_reference[index].code_payout is not None
            else code_pool
        )
        for index in sample_indices
    ]
    participation_multipliers = (
        [
            forecaster.sample_ticket_multiplier(rng)
            for _ in range(bootstrap_simulations)
        ]
        if forecaster is not None
        else []
    )

    def value_distribution(candidate_jackpot: float) -> tuple[float, list[float]]:
        if forecaster is None or forecast_date is None:
            point = lower_ev + probabilities[1] * candidate_jackpot * share_factor
            values = [
                sample_lower + probabilities[1] * candidate_jackpot * share_factor
                for sample_lower in sample_lower_values
            ]
            return point, values
        candidate = forecaster.forecast(candidate_jackpot, forecast_date, game)
        candidate_lambda = (
            candidate.estimated_tickets / total_outcomes() * popularity_factor
        )
        candidate_share = _poisson_share_factor(candidate_lambda)
        candidate_code_ev = code_pool / candidate.estimated_tickets
        point = (
            lower_ev
            + candidate_code_ev
            + probabilities[1] * candidate_jackpot * candidate_share
        )
        values = []
        for sample_lower, sample_pool, multiplier in zip(
            sample_lower_values,
            sample_code_pools,
            participation_multipliers,
            strict=True,
        ):
            sample_tickets = candidate.estimated_tickets * multiplier
            sample_lambda = sample_tickets / total_outcomes() * popularity_factor
            values.append(
                sample_lower
                + sample_pool / sample_tickets
                + probabilities[1]
                * candidate_jackpot
                * _poisson_share_factor(sample_lambda)
            )
        return point, values

    bootstrap_values = value_distribution(jackpot)[1]
    ci_low = _quantile(bootstrap_values, tail_probability)
    ci_high = _quantile(bootstrap_values, 1 - tail_probability)
    fair_jackpot = _solve_threshold(
        lambda candidate: value_distribution(candidate)[0], price, jackpot
    )
    assert fair_jackpot is not None
    conservative_jackpot = _solve_threshold(
        lambda candidate: _quantile(
            value_distribution(candidate)[1], tail_probability
        ),
        price,
        fair_jackpot,
    )
    support_max = participation.training_jackpot_max if participation else None
    return ValueReport(
        game=game,
        target_date=report_target_date,
        training_last_date=training_last_date,
        reference_draws=len(reference),
        jackpot=jackpot,
        expected_co_winners=expected_co_winners,
        expected_co_winners_source=co_winner_source,
        jackpot_share_factor=share_factor,
        estimated_tickets=(participation.estimated_tickets if participation else None),
        median_estimated_tickets=(
            participation.median_estimated_tickets if participation else None
        ),
        participation_smearing_factor=(participation.smearing_factor if participation else None),
        participation_model=(participation.model if participation else None),
        participation_log_rmse=(participation.backtest_log_rmse if participation else None),
        participation_uncertainty_method=(
            participation.uncertainty_method if participation else None
        ),
        participation_residual_observations=(
            participation.residual_observations if participation else None
        ),
        participation_jackpot_max=support_max,
        participation_extrapolated=(participation.extrapolated if participation else False),
        lower_rank_ev=lower_ev,
        lower_rank_ev_method="mean_of_draw_level_expected_values",
        predictive_payout_window=payout_window,
        predictive_tail_probability=tail_probability,
        payout_validation_coverage=payout_validation_coverage,
        prediction_interval_target=0.95,
        code_ev=code_ev,
        estimated_ev=estimated_ev,
        naive_ev=naive_ev,
        expected_return_rate=expected_return_rate,
        estimated_roi=expected_return_rate - 1,
        ev_ci_low=ci_low,
        ev_ci_high=ci_high,
        ticket_price=price,
        fair_jackpot=fair_jackpot,
        fair_jackpot_extrapolated=(
            support_max is not None and fair_jackpot > support_max
        ),
        conservative_jackpot=conservative_jackpot,
        conservative_jackpot_extrapolated=(
            support_max is not None
            and conservative_jackpot is not None
            and conservative_jackpot > support_max
        ),
        uncertainty_method="temporally_selected_payout_and_empirical_residual_bootstrap",
        decision="eligible" if ci_low > price else "no_bet",
        limitations=(
            "Le rang 9 fournit un proxy du nombre de grilles, pas le volume FDJ certifie.",
            "Le partage Poisson suppose une combinaison aussi populaire que la moyenne; "
            "ajuster popularity_factor pour un scenario anti-foule.",
            "Le calcul des codes suppose que leur pool historique se repartit sur les grilles.",
            "Les rapports futurs et le nombre reel de co-gagnants restent inconnus.",
            "Les residus de participation sont empiriques mais globaux, sans calibration locale.",
            "Les seuils reevaluent la participation mais deviennent des extrapolations au-dela "
            "du jackpot maximal observe.",
            "Un point positif ne suffit pas: la decision utilise la borne basse calibree vers "
            "95% de couverture historique.",
        ),
    )
