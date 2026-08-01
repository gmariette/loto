from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from math import exp, sqrt
from statistics import fmean

import numpy as np

from .domain import Draw
from .participation import fit_participation_forecaster, participation_observations
from .probability import rank_probabilities, total_outcomes
from .value import (
    TICKET_PRICES,
    _current_prize_draws,
    _lower_rank_ev,
    _mean_code_pool,
    _poisson_share_factor,
    _quantile,
    _select_predictive_reference,
)


@dataclass(frozen=True, slots=True)
class ValueBacktestResult:
    game: str
    observations: int
    folds: int
    first_test_date: str
    last_test_date: str
    fold_models: tuple[str, ...]
    fold_payout_windows: tuple[int, ...]
    mean_predicted_ev: float
    mean_observed_schedule_ev: float
    mean_bias: float
    mae: float
    rmse: float
    naive_mean_bias: float
    naive_mae: float
    naive_rmse: float
    relative_mae_improvement: float
    mae_delta: float
    mae_delta_ci_low: float
    mae_delta_ci_high: float
    permutation_p_value: float
    qualified_against_naive: bool
    prediction_interval_coverage: float
    coverage_ci_low: float
    coverage_ci_high: float
    prediction_interval_target: float
    mean_prediction_interval_width: float
    eligible_predictions: int
    observed_positive_schedules: int
    false_positive_decisions: int
    false_negative_decisions: int
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _draw_jackpot(draw: Draw) -> float | None:
    rank_1 = next((prize for prize in draw.prizes if prize.rank == 1), None)
    return rank_1.payout if rank_1 and rank_1.payout and rank_1.payout > 0 else None


def _draw_ticket_count(draw: Draw, rank_9_probability: float) -> float | None:
    rank_9 = next((prize for prize in draw.prizes if prize.rank == 9), None)
    if rank_9 is None or rank_9.winners is None or rank_9.winners <= 0:
        return None
    return rank_9.winners / rank_9_probability


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, center - margin)
    high = 1.0 if successes == total else min(1.0, center + margin)
    return low, high


def backtest_value(
    draws: list[Draw],
    game: str = "loto",
    min_train: int = 500,
    folds: int = 3,
    simulations: int = 500,
    seed: int = 0,
) -> ValueBacktestResult:
    if game not in TICKET_PRICES:
        raise ValueError(f"Jeu inconnu: {game}")
    if min_train < 200 or folds < 2 or simulations < 100:
        raise ValueError("Il faut min_train >= 200, 2 folds et 100 simulations")
    observations = participation_observations(draws)
    if len(observations) <= min_train:
        raise ValueError("Pas assez d'observations pour le backtest de valeur")
    dates = np.asarray([item.draw_date for item in observations], dtype=object)
    eligible_dates = np.unique(dates[min_train:])
    date_folds = [item for item in np.array_split(eligible_dates, folds) if len(item)]
    probabilities = {item.rank: item.probability for item in rank_probabilities()}
    rank_9_probability = probabilities[9]
    price = TICKET_PRICES[game]
    current_draws = _current_prize_draws(draws, game)
    rng = random.Random(seed)

    predicted_values: list[float] = []
    observed_values: list[float] = []
    naive_values: list[float] = []
    interval_lows: list[float] = []
    interval_highs: list[float] = []
    models: list[str] = []
    payout_windows: list[int] = []

    for fold_number, test_dates in enumerate(date_folds):
        first_test_date = test_dates[0]
        training_draws = [
            draw
            for draw in draws
            if draw.draw_date is not None and draw.draw_date < first_test_date
        ]
        training_count = len(participation_observations(training_draws))
        inner_min_train = max(100, training_count // 2)
        forecaster = fit_participation_forecaster(
            training_draws,
            min_train=inner_min_train,
            folds=2,
            simulations=simulations,
            seed=seed + fold_number * 1000,
        )
        models.append(forecaster.champion.model if forecaster.model is not None else "baseline")
        reference = _current_prize_draws(training_draws, game)
        if len(reference) < 5:
            continue
        lower_training = [_lower_rank_ev(draw, probabilities) for draw in reference]
        lower_mean = float(fmean(lower_training))
        code_pool_mean = _mean_code_pool(reference)
        predictive_reference, payout_window, _ = _select_predictive_reference(
            reference, probabilities
        )
        payout_windows.append(payout_window)
        lower_predictive = [
            _lower_rank_ev(draw, probabilities) for draw in predictive_reference
        ]
        code_pools = [
            draw.code_winners * draw.code_payout
            for draw in predictive_reference
            if draw.code_winners is not None and draw.code_payout is not None
        ]
        test_date_set = set(test_dates)
        targets = [draw for draw in current_draws if draw.draw_date in test_date_set]

        for target in targets:
            if target.draw_date is None:
                continue
            jackpot = _draw_jackpot(target)
            actual_tickets = _draw_ticket_count(target, rank_9_probability)
            if jackpot is None or actual_tickets is None:
                continue
            forecast = forecaster.forecast(jackpot, target.draw_date, game)
            expected_winners = forecast.estimated_tickets / total_outcomes()
            predicted_share = _poisson_share_factor(expected_winners)
            predicted_code_ev = code_pool_mean / forecast.estimated_tickets
            predicted_ev = (
                lower_mean + predicted_code_ev + probabilities[1] * jackpot * predicted_share
            )

            samples = []
            for _ in range(simulations):
                sample_tickets = forecast.estimated_tickets * exp(
                    rng.gauss(0, 1) * forecast.backtest_log_rmse
                    - 0.5 * forecast.backtest_log_rmse**2
                )
                sample_pool = rng.choice(code_pools) if code_pools else 0.0
                samples.append(
                    rng.choice(lower_predictive)
                    + sample_pool / sample_tickets
                    + probabilities[1]
                    * jackpot
                    * _poisson_share_factor(sample_tickets / total_outcomes())
                )

            actual_code_pool = (
                target.code_winners * target.code_payout
                if target.code_winners is not None and target.code_payout is not None
                else code_pool_mean
            )
            observed_ev = (
                _lower_rank_ev(target, probabilities)
                + actual_code_pool / actual_tickets
                + probabilities[1]
                * jackpot
                * _poisson_share_factor(actual_tickets / total_outcomes())
            )
            predicted_values.append(predicted_ev)
            naive_values.append(lower_mean + probabilities[1] * jackpot)
            observed_values.append(observed_ev)
            interval_lows.append(_quantile(samples, 0.025))
            interval_highs.append(_quantile(samples, 0.975))

    if not predicted_values:
        raise ValueError(f"Aucun tirage {game} evaluable dans les folds")
    errors = [
        predicted - observed
        for predicted, observed in zip(predicted_values, observed_values, strict=True)
    ]
    naive_errors = [
        predicted - observed
        for predicted, observed in zip(naive_values, observed_values, strict=True)
    ]
    mae = float(fmean(abs(error) for error in errors))
    naive_mae = float(fmean(abs(error) for error in naive_errors))
    absolute_error_deltas = np.asarray(
        [abs(error) - abs(naive) for error, naive in zip(errors, naive_errors, strict=True)]
    )
    significance_rng = np.random.default_rng(seed + 50_000)
    bootstrap_deltas = np.asarray(
        [
            significance_rng.choice(
                absolute_error_deltas, size=len(absolute_error_deltas), replace=True
            ).mean()
            for _ in range(simulations)
        ]
    )
    observed_delta = float(absolute_error_deltas.mean())
    extreme = 0
    for _ in range(simulations):
        signs = significance_rng.choice((-1.0, 1.0), size=len(absolute_error_deltas))
        if float((absolute_error_deltas * signs).mean()) <= observed_delta:
            extreme += 1
    p_value = (extreme + 1) / (simulations + 1)
    delta_low = float(np.quantile(bootstrap_deltas, 0.025))
    delta_high = float(np.quantile(bootstrap_deltas, 0.975))
    coverage = [
        low <= observed <= high
        for observed, low, high in zip(
            observed_values, interval_lows, interval_highs, strict=True
        )
    ]
    eligible = [low > price for low in interval_lows]
    positives = [observed > price for observed in observed_values]
    coverage_low, coverage_high = _wilson_interval(sum(coverage), len(coverage))
    return ValueBacktestResult(
        game=game,
        observations=len(predicted_values),
        folds=len(date_folds),
        first_test_date=min(
            draw.draw_date for draw in current_draws if draw.draw_date is not None
            and draw.draw_date in set(eligible_dates)
        ).isoformat(),
        last_test_date=max(
            draw.draw_date for draw in current_draws if draw.draw_date is not None
            and draw.draw_date in set(eligible_dates)
        ).isoformat(),
        fold_models=tuple(models),
        fold_payout_windows=tuple(payout_windows),
        mean_predicted_ev=float(fmean(predicted_values)),
        mean_observed_schedule_ev=float(fmean(observed_values)),
        mean_bias=float(fmean(errors)),
        mae=mae,
        rmse=sqrt(fmean(error**2 for error in errors)),
        naive_mean_bias=float(fmean(naive_errors)),
        naive_mae=naive_mae,
        naive_rmse=sqrt(fmean(error**2 for error in naive_errors)),
        relative_mae_improvement=1 - mae / naive_mae,
        mae_delta=observed_delta,
        mae_delta_ci_low=delta_low,
        mae_delta_ci_high=delta_high,
        permutation_p_value=p_value,
        qualified_against_naive=delta_high < 0 and p_value < 0.05,
        prediction_interval_coverage=sum(coverage) / len(coverage),
        coverage_ci_low=coverage_low,
        coverage_ci_high=coverage_high,
        prediction_interval_target=0.95,
        mean_prediction_interval_width=float(
            fmean(high - low for low, high in zip(interval_lows, interval_highs, strict=True))
        ),
        eligible_predictions=sum(eligible),
        observed_positive_schedules=sum(positives),
        false_positive_decisions=sum(
            predicted and not observed
            for predicted, observed in zip(eligible, positives, strict=True)
        ),
        false_negative_decisions=sum(
            not predicted and observed
            for predicted, observed in zip(eligible, positives, strict=True)
        ),
        limitations=(
            "La cible est une esperance reconstruite depuis le bareme, pas le gain d'une grille.",
            "Le partage observe reste approxime par un modele Poisson de popularite moyenne.",
            "Les folds bloques n'actualisent pas le modele a l'interieur de leur periode test.",
        ),
    )
