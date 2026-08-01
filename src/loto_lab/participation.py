from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date
from math import exp, log
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .domain import Draw
from .probability import rank_probabilities

GAMES = ("loto", "super_loto", "grand_loto")
MODEL_PARAMETERS: dict[str, tuple[dict[str, float | int], ...]] = {
    "ridge": ({"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 100.0}),
    "gradient_boosting": (
        {"learning_rate": 0.03, "max_leaf_nodes": 7, "l2": 5.0},
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2": 5.0},
    ),
}


@dataclass(frozen=True, slots=True)
class ParticipationObservation:
    draw_date: date
    game: str
    jackpot: float
    rank_9_winners: int
    estimated_tickets: float


@dataclass(frozen=True, slots=True)
class ParticipationBacktestResult:
    model: str
    observations: int
    test_observations: int
    folds: int
    selected_parameters: tuple[dict[str, float | int], ...]
    final_parameters: dict[str, float | int]
    log_rmse: float
    baseline_log_rmse: float
    mape: float
    baseline_mape: float
    relative_rmse_improvement: float
    mse_delta: float
    delta_ci_low: float
    delta_ci_high: float
    permutation_p_value: float
    adjusted_p_value: float
    calibration_factors: tuple[float, ...]
    raw_level_bias_rate: float
    calibrated_level_bias_rate: float
    qualified: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParticipationForecast:
    game: str
    target_date: date
    jackpot: float
    observations: int
    model: str
    parameters: dict[str, float | int]
    median_estimated_tickets: float
    estimated_tickets: float
    smearing_factor: float
    backtest_log_rmse: float
    baseline_log_rmse: float
    training_jackpot_min: float
    training_jackpot_max: float
    extrapolated: bool
    uncertainty_method: str
    residual_observations: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ParticipationForecaster:
    observations: tuple[ParticipationObservation, ...]
    champion: ParticipationBacktestResult
    model: Any | None
    smearing_factor: float
    ticket_multipliers: tuple[float, ...]

    def forecast(
        self, jackpot: float, target_date: date, game: str = "loto"
    ) -> ParticipationForecast:
        if jackpot <= 0:
            raise ValueError("Le jackpot doit etre positif")
        last_training_date = max(item.draw_date for item in self.observations)
        if target_date <= last_training_date:
            raise ValueError(
                "La date cible doit etre posterieure aux observations d'apprentissage"
            )
        target = ParticipationObservation(target_date, game, jackpot, 0, 0)
        observations = list(self.observations)
        if self.model is not None:
            log_estimate = float(
                self.model.predict(_features([target], observations[0].draw_date))[0]
            )
            median_estimate = exp(log_estimate)
            model_name = self.champion.model
            parameters = self.champion.final_parameters
        else:
            median_estimate = float(exp(_baseline_predictions(observations, [target])[0]))
            model_name = "segmented_median"
            parameters = {}
        game_jackpots = [item.jackpot for item in observations if item.game == game]
        support = game_jackpots or [item.jackpot for item in observations]
        support_min = min(support)
        support_max = max(support)
        return ParticipationForecast(
            game,
            target_date,
            jackpot,
            len(observations),
            model_name,
            parameters,
            median_estimate,
            median_estimate * self.smearing_factor,
            self.smearing_factor,
            self.champion.log_rmse,
            self.champion.baseline_log_rmse,
            support_min,
            support_max,
            jackpot < support_min or jackpot > support_max,
            "temporal_empirical_residuals",
            len(self.ticket_multipliers),
        )

    def sample_ticket_multiplier(self, rng: Any) -> float:
        return float(rng.choice(self.ticket_multipliers))


def participation_observations(draws: list[Draw]) -> list[ParticipationObservation]:
    rank_9_probability = {item.rank: item.probability for item in rank_probabilities()}[9]
    observations = []
    for draw in draws:
        prizes = {prize.rank: prize for prize in draw.prizes}
        rank_1 = prizes.get(1)
        rank_9 = prizes.get(9)
        if (
            draw.draw_date is None
            or rank_1 is None
            or rank_1.payout is None
            or rank_1.payout <= 0
            or rank_9 is None
            or rank_9.winners is None
            or rank_9.winners <= 0
        ):
            continue
        observations.append(
            ParticipationObservation(
                draw.draw_date,
                draw.game,
                _advertised_jackpot(draw),
                rank_9.winners,
                rank_9.winners / rank_9_probability,
            )
        )
    return sorted(observations, key=lambda item: (item.draw_date, item.game))


def _advertised_jackpot(draw: Draw) -> float:
    rank_1 = next(prize for prize in draw.prizes if prize.rank == 1)
    assert rank_1.payout is not None
    winners = rank_1.winners or 0
    return rank_1.payout * winners if winners > 1 else rank_1.payout


def _features(
    observations: list[ParticipationObservation], reference_date: date
) -> np.ndarray:
    rows = []
    for observation in observations:
        jackpot = log(1 + observation.jackpot) / 20
        weekday = observation.draw_date.weekday()
        month_angle = 2 * np.pi * observation.draw_date.month / 12
        rows.append(
            [
                jackpot,
                jackpot**2,
                (observation.draw_date - reference_date).days / 3652,
                np.sin(month_angle),
                np.cos(month_angle),
                *(float(weekday == value) for value in range(7)),
                *(float(observation.game == game) for game in GAMES),
            ]
        )
    return np.asarray(rows, dtype=float)


def _fit_model(
    model_name: str,
    parameters: dict[str, float | int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Any:
    if model_name == "ridge":
        model: Any = Ridge(alpha=float(parameters["alpha"]))
    elif model_name == "gradient_boosting":
        model = HistGradientBoostingRegressor(
            learning_rate=float(parameters["learning_rate"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            l2_regularization=float(parameters["l2"]),
            max_iter=150,
            random_state=seed,
        )
    else:
        raise ValueError(f"Modele de participation inconnu: {model_name}")
    return model.fit(x_train, y_train)


def _baseline_predictions(
    train: list[ParticipationObservation], test: list[ParticipationObservation]
) -> np.ndarray:
    by_game_weekday: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_game: dict[str, list[float]] = defaultdict(list)
    all_values = []
    for observation in train:
        value = log(observation.estimated_tickets)
        by_game_weekday[(observation.game, observation.draw_date.weekday())].append(value)
        by_game[observation.game].append(value)
        all_values.append(value)
    predictions = []
    for observation in test:
        specific = by_game_weekday[(observation.game, observation.draw_date.weekday())]
        game_values = by_game[observation.game]
        values = specific if len(specific) >= 10 else game_values if game_values else all_values
        predictions.append(float(np.median(values)))
    return np.asarray(predictions)


def _select_parameters(
    observations: list[ParticipationObservation],
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    model_name: str,
    seed: int,
) -> dict[str, float | int]:
    dates = np.asarray([item.draw_date for item in observations], dtype=object)
    train_dates = np.unique(dates[train_indices])
    validation_dates = train_dates[max(1, int(len(train_dates) * 0.8)) :]
    if not len(validation_dates):
        return dict(MODEL_PARAMETERS[model_name][0])
    split_date = validation_dates[0]
    inner_train = train_indices[dates[train_indices] < split_date]
    validation = train_indices[dates[train_indices] >= split_date]
    best: tuple[float, dict[str, float | int]] | None = None
    for parameters in MODEL_PARAMETERS[model_name]:
        model = _fit_model(model_name, parameters, x[inner_train], y[inner_train], seed)
        score = float(np.mean((model.predict(x[validation]) - y[validation]) ** 2))
        if best is None or score < best[0]:
            best = (score, dict(parameters))
    assert best is not None
    return best[1]


def _smearing_factor(
    observations: list[ParticipationObservation],
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    model_name: str,
    parameters: dict[str, float | int],
    seed: int,
) -> float:
    dates = np.asarray([item.draw_date for item in observations], dtype=object)
    train_dates = np.unique(dates[train_indices])
    validation_dates = train_dates[max(1, int(len(train_dates) * 0.8)) :]
    if not len(validation_dates):
        return 1.0
    split_date = validation_dates[0]
    inner_train = train_indices[dates[train_indices] < split_date]
    validation = train_indices[dates[train_indices] >= split_date]
    model = _fit_model(model_name, parameters, x[inner_train], y[inner_train], seed)
    residuals = y[validation] - model.predict(x[validation])
    return float(np.mean(np.exp(residuals)))


def participation_backtest(
    draws: list[Draw],
    min_train: int = 500,
    folds: int = 3,
    simulations: int = 2_000,
    seed: int = 0,
) -> list[ParticipationBacktestResult]:
    observations = participation_observations(draws)
    if min_train < 100 or folds < 2 or simulations < 100 or len(observations) <= min_train:
        raise ValueError("Pas assez d'observations pour le backtest de participation")
    reference_date = observations[0].draw_date
    x = _features(observations, reference_date)
    y = np.log([item.estimated_tickets for item in observations])
    dates = np.asarray([item.draw_date for item in observations], dtype=object)
    eligible_dates = np.unique(dates[min_train:])
    date_folds = [item for item in np.array_split(eligible_dates, folds) if len(item)]
    results = []
    for model_number, model_name in enumerate(MODEL_PARAMETERS):
        predictions = []
        baselines = []
        actuals = []
        selected = []
        calibration_factors = []
        fold_calibration_factors = []
        for fold_number, test_dates in enumerate(date_folds):
            train_indices = np.flatnonzero(dates < test_dates[0])
            test_indices = np.flatnonzero(np.isin(dates, test_dates))
            parameters = _select_parameters(
                observations,
                x,
                y,
                train_indices,
                model_name,
                seed + model_number * 1000 + fold_number,
            )
            model = _fit_model(
                model_name,
                parameters,
                x[train_indices],
                y[train_indices],
                seed + model_number * 1000 + fold_number,
            )
            calibration_factor = _smearing_factor(
                observations,
                x,
                y,
                train_indices,
                model_name,
                parameters,
                seed + model_number * 1000 + fold_number,
            )
            predictions.extend(model.predict(x[test_indices]))
            baselines.extend(
                _baseline_predictions(
                    [observations[index] for index in train_indices],
                    [observations[index] for index in test_indices],
                )
            )
            actuals.extend(y[test_indices])
            selected.append(parameters)
            calibration_factors.extend([calibration_factor] * len(test_indices))
            fold_calibration_factors.append(calibration_factor)
        predicted = np.asarray(predictions)
        baseline = np.asarray(baselines)
        actual = np.asarray(actuals)
        calibration = np.asarray(calibration_factors)
        rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
        baseline_rmse = float(np.sqrt(np.mean((baseline - actual) ** 2)))
        deltas = (predicted - actual) ** 2 - (baseline - actual) ** 2
        rng = np.random.default_rng(seed + model_number * 10_000)
        bootstrap = np.asarray(
            [rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(simulations)]
        )
        observed_delta = float(deltas.mean())
        extreme = 0
        for _ in range(simulations):
            signs = rng.choice((-1.0, 1.0), size=len(deltas))
            if float((deltas * signs).mean()) <= observed_delta:
                extreme += 1
        p_value = (extreme + 1) / (simulations + 1)
        final_parameters = _select_parameters(
            observations, x, y, np.arange(len(observations)), model_name, seed + folds
        )
        raw_levels = np.exp(predicted)
        calibrated_levels = raw_levels * calibration
        actual_levels = np.exp(actual)
        results.append(
            ParticipationBacktestResult(
                model_name,
                len(observations),
                len(actual),
                len(date_folds),
                tuple(selected),
                final_parameters,
                rmse,
                baseline_rmse,
                float(np.mean(np.abs(np.exp(predicted - actual) - 1))),
                float(np.mean(np.abs(np.exp(baseline - actual) - 1))),
                1 - rmse / baseline_rmse,
                observed_delta,
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
                p_value,
                1.0,
                tuple(fold_calibration_factors),
                float((raw_levels.sum() - actual_levels.sum()) / actual_levels.sum()),
                float(
                    (calibrated_levels.sum() - actual_levels.sum()) / actual_levels.sum()
                ),
                False,
            )
        )
    order = sorted(range(len(results)), key=lambda index: results[index].permutation_p_value)
    adjusted = [1.0] * len(results)
    running = 0.0
    for position, index in enumerate(order):
        value = min(1.0, (len(results) - position) * results[index].permutation_p_value)
        running = max(running, value)
        adjusted[index] = running
    return [
        replace(
            result,
            adjusted_p_value=adjusted[index],
            qualified=result.delta_ci_high < 0 and adjusted[index] < 0.05,
        )
        for index, result in enumerate(results)
    ]


def _temporal_ticket_multipliers(
    observations: list[ParticipationObservation],
    champion: ParticipationBacktestResult,
    min_train: int,
    folds: int,
    seed: int,
) -> tuple[float, ...]:
    reference_date = observations[0].draw_date
    x = _features(observations, reference_date)
    y = np.log([item.estimated_tickets for item in observations])
    dates = np.asarray([item.draw_date for item in observations], dtype=object)
    eligible_dates = np.unique(dates[min_train:])
    date_folds = [item for item in np.array_split(eligible_dates, folds) if len(item)]
    residuals = []
    model_seed_offset = (
        list(MODEL_PARAMETERS).index(champion.model) * 1000 if champion.qualified else 0
    )
    for fold_number, test_dates in enumerate(date_folds):
        train_indices = np.flatnonzero(dates < test_dates[0])
        test_indices = np.flatnonzero(np.isin(dates, test_dates))
        if champion.qualified:
            model = _fit_model(
                champion.model,
                champion.selected_parameters[fold_number],
                x[train_indices],
                y[train_indices],
                seed + model_seed_offset + fold_number,
            )
            predicted = model.predict(x[test_indices])
        else:
            predicted = _baseline_predictions(
                [observations[index] for index in train_indices],
                [observations[index] for index in test_indices],
            )
        residuals.extend(y[test_indices] - predicted)
    level_errors = np.exp(np.asarray(residuals))
    normalized = level_errors / level_errors.mean()
    return tuple(float(value) for value in normalized)


def fit_participation_forecaster(
    draws: list[Draw],
    min_train: int = 500,
    folds: int = 3,
    simulations: int = 2_000,
    seed: int = 0,
) -> ParticipationForecaster:
    observations = participation_observations(draws)
    results = participation_backtest(draws, min_train, folds, simulations, seed)
    qualified = [result for result in results if result.qualified]
    champion = min(qualified or results, key=lambda result: result.log_rmse)
    if qualified:
        x = _features(observations, observations[0].draw_date)
        y = np.log([item.estimated_tickets for item in observations])
        indices = np.arange(len(observations))
        final_seed = seed + folds
        calibration = _smearing_factor(
            observations,
            x,
            y,
            indices,
            champion.model,
            champion.final_parameters,
            final_seed,
        )
        model = _fit_model(champion.model, champion.final_parameters, x, y, final_seed)
    else:
        calibration = 1.0
        model = None
    multipliers = _temporal_ticket_multipliers(
        observations, champion, min_train, folds, seed
    )
    return ParticipationForecaster(
        tuple(observations), champion, model, calibration, multipliers
    )


def forecast_participation(
    draws: list[Draw],
    jackpot: float,
    target_date: date,
    game: str = "loto",
    min_train: int = 500,
    folds: int = 3,
    simulations: int = 2_000,
    seed: int = 0,
) -> ParticipationForecast:
    historical = [
        draw
        for draw in draws
        if draw.draw_date is not None and draw.draw_date < target_date
    ]
    forecaster = fit_participation_forecaster(historical, min_train, folds, simulations, seed)
    return forecaster.forecast(jackpot, target_date, game)
