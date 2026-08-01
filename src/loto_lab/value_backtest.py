from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from math import ceil, sqrt
from statistics import fmean

import numpy as np

from .domain import Draw
from .participation import (
    _advertised_jackpot,
    fit_participation_forecaster,
    participation_observations,
)
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
class ValueBacktestPeriod:
    period: int
    fold: int
    training_last_date: str
    first_test_date: str
    last_test_date: str
    observations: int
    participation_model: str
    payout_window: int
    tail_probability: float
    mean_bias: float
    mae: float
    naive_mae: float
    relative_mae_improvement: float
    prediction_interval_coverage: float
    eligible_predictions: int
    false_positive_decisions: int


@dataclass(frozen=True, slots=True)
class ValueBacktestResult:
    game: str
    observations: int
    folds: int
    first_test_date: str
    last_test_date: str
    fold_models: tuple[str, ...]
    fold_payout_windows: tuple[int, ...]
    fold_tail_probabilities: tuple[float, ...]
    refit_interval_dates: int
    refits: int
    periods_better_than_naive: int
    periods: tuple[ValueBacktestPeriod, ...]
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
    coverage_compatible_with_target: bool
    value_model_qualified: bool
    inference_method: str
    temporal_block_size: int
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
    return _advertised_jackpot(draw) if rank_1 and rank_1.payout and rank_1.payout > 0 else None


def _draw_ticket_count(draw: Draw, rank_9_probability: float) -> float | None:
    rank_9 = next((prize for prize in draw.prizes if prize.rank == 9), None)
    if rank_9 is None or rank_9.winners is None or rank_9.winners <= 0:
        return None
    return rank_9.winners / rank_9_probability


def _moving_block_means(
    values: np.ndarray,
    simulations: int,
    rng: np.random.Generator,
    block_size: int,
) -> np.ndarray:
    size = len(values)
    width = min(block_size, size)
    blocks_needed = ceil(size / width)
    means = []
    for _ in range(simulations):
        starts = rng.integers(0, size - width + 1, size=blocks_needed)
        sample = np.concatenate([values[start : start + width] for start in starts])[:size]
        means.append(float(sample.mean()))
    return np.asarray(means)


def _period_result(
    period: int,
    fold: int,
    training_last_date: str,
    test_dates: list[str],
    model: str,
    payout_window: int,
    tail_probability: float,
    predicted: list[float],
    observed: list[float],
    naive: list[float],
    interval_lows: list[float],
    interval_highs: list[float],
    price: float,
) -> ValueBacktestPeriod:
    errors = [estimate - actual for estimate, actual in zip(predicted, observed, strict=True)]
    mae = float(fmean(abs(error) for error in errors))
    naive_mae = float(
        fmean(abs(estimate - actual) for estimate, actual in zip(naive, observed, strict=True))
    )
    eligible = [low > price for low in interval_lows]
    positives = [actual > price for actual in observed]
    coverage = sum(
        low <= actual <= high
        for actual, low, high in zip(observed, interval_lows, interval_highs, strict=True)
    ) / len(observed)
    return ValueBacktestPeriod(
        period=period,
        fold=fold,
        training_last_date=training_last_date,
        first_test_date=min(test_dates),
        last_test_date=max(test_dates),
        observations=len(observed),
        participation_model=model,
        payout_window=payout_window,
        tail_probability=tail_probability,
        mean_bias=float(fmean(errors)),
        mae=mae,
        naive_mae=naive_mae,
        relative_mae_improvement=1 - mae / naive_mae,
        prediction_interval_coverage=coverage,
        eligible_predictions=sum(eligible),
        false_positive_decisions=sum(
            decision and not positive
            for decision, positive in zip(eligible, positives, strict=True)
        ),
    )


def backtest_value(
    draws: list[Draw],
    game: str = "loto",
    min_train: int = 500,
    folds: int = 3,
    simulations: int = 500,
    seed: int = 0,
    block_size: int = 12,
    refit_interval: int = 52,
) -> ValueBacktestResult:
    if game not in TICKET_PRICES:
        raise ValueError(f"Jeu inconnu: {game}")
    if (
        min_train < 200
        or folds < 2
        or simulations < 100
        or block_size < 1
        or refit_interval < 1
    ):
        raise ValueError(
            "Il faut min_train >= 200, 2 folds, 100 simulations, block_size >= 1 "
            "et refit_interval >= 1"
        )
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
    fold_models: list[str] = []
    fold_payout_windows: list[int] = []
    fold_tail_probabilities: list[float] = []
    periods: list[ValueBacktestPeriod] = []
    refit_number = 0

    for fold_number, test_dates in enumerate(date_folds):
        fold_recorded = False
        for batch_start in range(0, len(test_dates), refit_interval):
            batch_dates = test_dates[batch_start : batch_start + refit_interval]
            test_date_set = set(batch_dates)
            targets = [draw for draw in current_draws if draw.draw_date in test_date_set]
            if not targets:
                continue
            first_test_date = min(
                draw.draw_date for draw in targets if draw.draw_date is not None
            )
            training_draws = [
                draw
                for draw in draws
                if draw.draw_date is not None and draw.draw_date < first_test_date
            ]
            reference = _current_prize_draws(training_draws, game)
            if len(reference) < 5:
                continue
            training_count = len(participation_observations(training_draws))
            inner_min_train = max(100, training_count // 2)
            forecaster = fit_participation_forecaster(
                training_draws,
                min_train=inner_min_train,
                folds=2,
                simulations=simulations,
                seed=seed + refit_number * 1000,
            )
            model_name = (
                forecaster.champion.model if forecaster.model is not None else "baseline"
            )
            lower_training = [_lower_rank_ev(draw, probabilities) for draw in reference]
            lower_mean = float(fmean(lower_training))
            code_pool_mean = _mean_code_pool(reference)
            predictive_reference, payout_window, _, tail_probability = (
                _select_predictive_reference(reference, probabilities)
            )
            if not fold_recorded:
                fold_models.append(model_name)
                fold_payout_windows.append(payout_window)
                fold_tail_probabilities.append(tail_probability)
                fold_recorded = True
            lower_predictive = [
                _lower_rank_ev(draw, probabilities) for draw in predictive_reference
            ]
            code_pools = [
                draw.code_winners * draw.code_payout
                for draw in predictive_reference
                if draw.code_winners is not None and draw.code_payout is not None
            ]
            period_start = len(predicted_values)
            evaluated_dates: list[str] = []

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
                    lower_mean
                    + predicted_code_ev
                    + probabilities[1] * jackpot * predicted_share
                )

                samples = []
                for _ in range(simulations):
                    sample_tickets = (
                        forecast.estimated_tickets
                        * forecaster.sample_ticket_multiplier(rng)
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
                interval_lows.append(_quantile(samples, tail_probability))
                interval_highs.append(_quantile(samples, 1 - tail_probability))
                evaluated_dates.append(target.draw_date.isoformat())

            if len(predicted_values) > period_start:
                periods.append(
                    _period_result(
                        period=len(periods) + 1,
                        fold=fold_number + 1,
                        training_last_date=max(
                            draw.draw_date
                            for draw in training_draws
                            if draw.draw_date is not None
                        ).isoformat(),
                        test_dates=evaluated_dates,
                        model=model_name,
                        payout_window=payout_window,
                        tail_probability=tail_probability,
                        predicted=predicted_values[period_start:],
                        observed=observed_values[period_start:],
                        naive=naive_values[period_start:],
                        interval_lows=interval_lows[period_start:],
                        interval_highs=interval_highs[period_start:],
                        price=price,
                    )
                )
                refit_number += 1

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
    effective_block_size = min(block_size, len(absolute_error_deltas))
    bootstrap_deltas = _moving_block_means(
        absolute_error_deltas,
        simulations,
        significance_rng,
        effective_block_size,
    )
    observed_delta = float(absolute_error_deltas.mean())
    extreme = 0
    for _ in range(simulations):
        signs = np.repeat(
            significance_rng.choice(
                (-1.0, 1.0),
                size=ceil(len(absolute_error_deltas) / effective_block_size),
            ),
            effective_block_size,
        )[: len(absolute_error_deltas)]
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
    coverage_rate = sum(coverage) / len(coverage)
    coverage_bootstrap = _moving_block_means(
        np.asarray(coverage, dtype=float),
        simulations,
        significance_rng,
        effective_block_size,
    )
    coverage_low = min(coverage_rate, float(np.quantile(coverage_bootstrap, 0.025)))
    coverage_high = max(coverage_rate, float(np.quantile(coverage_bootstrap, 0.975)))
    qualified_against_naive = delta_high < 0 and p_value < 0.05
    coverage_compatible = coverage_low <= 0.95 <= coverage_high
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
        fold_models=tuple(fold_models),
        fold_payout_windows=tuple(fold_payout_windows),
        fold_tail_probabilities=tuple(fold_tail_probabilities),
        refit_interval_dates=refit_interval,
        refits=len(periods),
        periods_better_than_naive=sum(period.mae < period.naive_mae for period in periods),
        periods=tuple(periods),
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
        qualified_against_naive=qualified_against_naive,
        coverage_compatible_with_target=coverage_compatible,
        value_model_qualified=qualified_against_naive and coverage_compatible,
        inference_method="moving_block_bootstrap_and_block_sign_permutation",
        temporal_block_size=effective_block_size,
        prediction_interval_coverage=coverage_rate,
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
            f"Le modele est actualise toutes les {refit_interval} dates, pas apres chaque tirage.",
        ),
    )
