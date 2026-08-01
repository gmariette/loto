from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import date
from itertools import groupby
from math import cos, log1p, pi, sin
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .domain import DEFAULT_RULES, Draw, LotteryRules

FEATURE_NAMES = (
    "history_draws",
    "all_count",
    "all_frequency_delta",
    "frequency_10_delta",
    "frequency_50_delta",
    "frequency_200_delta",
    "normalized_gap",
    "present_previous_date",
    "previous_pair_affinity",
    "weekday_sin",
    "weekday_cos",
    "game_super_loto",
    "game_grand_loto",
    "year_trend",
) + tuple(f"number_{number}" for number in range(1, 50))


@dataclass(frozen=True, slots=True)
class FeatureDataset:
    x: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    draws: tuple[Draw, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @property
    def draw_count(self) -> int:
        return self.x.shape[0]


class _RollingCounts:
    def __init__(self, window: int, pool: int) -> None:
        self.window = window
        self.values: deque[tuple[int, ...]] = deque()
        self.counts = np.zeros(pool, dtype=float)

    def add(self, numbers: tuple[int, ...]) -> None:
        if len(self.values) == self.window:
            removed = self.values.popleft()
            self.counts[np.asarray(removed) - 1] -= 1
        self.values.append(numbers)
        self.counts[np.asarray(numbers) - 1] += 1

    def frequency(self, index: int) -> float:
        return self.counts[index] / len(self.values) if self.values else 0.0


class _FeatureState:
    def __init__(self, rules: LotteryRules = DEFAULT_RULES) -> None:
        self.rules = rules
        self.history_draws = 0
        self.counts = np.zeros(rules.main_pool, dtype=float)
        self.last_seen = np.full(rules.main_pool, -1, dtype=int)
        self.pair_counts = np.zeros((rules.main_pool, rules.main_pool), dtype=float)
        self.rolling = [_RollingCounts(window, rules.main_pool) for window in (10, 50, 200)]
        self.previous_date_numbers: set[int] = set()

    def matrix(self, target: Draw) -> np.ndarray:
        pool = self.rules.main_pool
        expected = self.rules.main_drawn / pool
        conditional_expected = (self.rules.main_drawn - 1) / (pool - 1)
        weekday = target.draw_date.weekday() if target.draw_date else 0
        year = target.draw_date.year if target.draw_date else 2008
        matrix = np.zeros((pool, len(FEATURE_NAMES)), dtype=float)

        for index in range(pool):
            number = index + 1
            count = self.counts[index]
            frequency = count / self.history_draws if self.history_draws else expected
            if self.last_seen[index] < 0:
                gap = self.history_draws + 1
            else:
                gap = self.history_draws - self.last_seen[index]
            gap_scale = log1p(max(1, self.history_draws))

            affinities = []
            for previous in self.previous_date_numbers:
                previous_count = self.counts[previous - 1]
                if previous_count:
                    affinities.append(
                        self.pair_counts[index, previous - 1] / previous_count
                        - conditional_expected
                    )

            values = (
                float(self.history_draws),
                count,
                frequency - expected,
                self.rolling[0].frequency(index) - expected,
                self.rolling[1].frequency(index) - expected,
                self.rolling[2].frequency(index) - expected,
                log1p(gap) / gap_scale,
                float(number in self.previous_date_numbers),
                sum(affinities) / len(affinities) if affinities else 0.0,
                sin(2 * pi * weekday / 7),
                cos(2 * pi * weekday / 7),
                float(target.game == "super_loto"),
                float(target.game == "grand_loto"),
                (year - 2008) / 20,
            )
            matrix[index, : len(values)] = values
            matrix[index, len(values) + index] = 1.0
        return matrix

    def update_date_group(self, draws: list[Draw]) -> None:
        current_date_numbers: set[int] = set()
        for draw in draws:
            numbers = np.asarray(draw.main) - 1
            self.counts[numbers] += 1
            for left in numbers:
                for right in numbers:
                    if left != right:
                        self.pair_counts[left, right] += 1
            for index in numbers:
                self.last_seen[index] = self.history_draws
            for rolling in self.rolling:
                rolling.add(draw.main)
            current_date_numbers.update(draw.main)
            self.history_draws += 1
        self.previous_date_numbers = current_date_numbers


def build_feature_dataset(
    draws: list[Draw], min_history: int = 50, rules: LotteryRules = DEFAULT_RULES
) -> FeatureDataset:
    if min_history < 1:
        raise ValueError("min_history doit etre positif")
    ordered = sorted(draws, key=lambda draw: (draw.draw_date is None, draw.draw_date, draw.game))
    state = _FeatureState(rules)
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    dates: list[int] = []
    targets: list[Draw] = []

    for _, iterator in groupby(ordered, key=lambda draw: draw.draw_date):
        group = list(iterator)
        if state.history_draws >= min_history:
            for draw in group:
                matrices.append(state.matrix(draw))
                actual = np.zeros(rules.main_pool, dtype=float)
                actual[np.asarray(draw.main) - 1] = 1.0
                labels.append(actual)
                dates.append(draw.draw_date.toordinal() if draw.draw_date else state.history_draws)
                targets.append(draw)
        state.update_date_group(group)
    if not matrices:
        raise ValueError("Historique insuffisant pour construire les variables")
    return FeatureDataset(
        np.stack(matrices),
        np.stack(labels),
        np.asarray(dates, dtype=int),
        tuple(targets),
    )


def next_feature_matrix(
    draws: list[Draw], target_date: date, game: str = "loto",
    rules: LotteryRules = DEFAULT_RULES,
) -> np.ndarray:
    state = _FeatureState(rules)
    ordered = sorted(draws, key=lambda draw: (draw.draw_date is None, draw.draw_date, draw.game))
    for _, iterator in groupby(ordered, key=lambda draw: draw.draw_date):
        state.update_date_group(list(iterator))
    placeholder = Draw(tuple(range(1, rules.main_drawn + 1)), 1, target_date, game)
    return state.matrix(placeholder)


def project_inclusion_probabilities(
    probabilities: np.ndarray, total: int = 5
) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    logits = np.log(clipped / (1 - clipped))
    low, high = -30.0, 30.0
    for _ in range(80):
        shift = (low + high) / 2
        projected = 1 / (1 + np.exp(-(logits + shift)))
        if projected.sum() > total:
            high = shift
        else:
            low = shift
    return 1 / (1 + np.exp(-(logits + (low + high) / 2)))


MODEL_PARAMETERS: dict[str, tuple[dict[str, float | int], ...]] = {
    "bayesian": (
        {"prior_strength": 50.0},
        {"prior_strength": 200.0},
        {"prior_strength": 1000.0},
    ),
    "logistic": ({"c": 0.01}, {"c": 0.1}, {"c": 1.0}),
    "gradient_boosting": (
        {"learning_rate": 0.03, "max_leaf_nodes": 7, "l2": 1.0},
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2": 5.0},
    ),
}

UNIFORM_BLEND_WEIGHTS = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)


def blend_with_uniform(probabilities: np.ndarray, weight: float) -> np.ndarray:
    if not 0 <= weight <= 1:
        raise ValueError("Le poids du modele doit etre compris entre 0 et 1")
    baseline = np.full_like(probabilities, 5 / 49, dtype=float)
    return baseline + weight * (probabilities - baseline)


def _fit_predict(
    model_name: str,
    parameters: dict[str, float | int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    blend_weight = float(parameters.get("uniform_blend", 1.0))
    test_shape = x_test.shape[:2]
    if model_name == "bayesian":
        history = x_test[:, :, 0]
        counts = x_test[:, :, 1]
        prior = float(parameters["prior_strength"])
        raw = (counts + prior * (5 / 49)) / (history + prior)
    else:
        train_x = x_train.reshape(-1, x_train.shape[-1])
        train_y = y_train.reshape(-1)
        test_x = x_test.reshape(-1, x_test.shape[-1])
        if model_name == "logistic":
            estimator: Any = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=float(parameters["c"]),
                    max_iter=500,
                    solver="lbfgs",
                    random_state=seed,
                ),
            )
        elif model_name == "gradient_boosting":
            estimator = HistGradientBoostingClassifier(
                learning_rate=float(parameters["learning_rate"]),
                max_leaf_nodes=int(parameters["max_leaf_nodes"]),
                l2_regularization=float(parameters["l2"]),
                max_iter=100,
                random_state=seed,
            )
        else:
            raise ValueError(f"Modele inconnu: {model_name}")
        estimator.fit(train_x, train_y)
        raw = estimator.predict_proba(test_x)[:, 1].reshape(test_shape)
    projected = np.stack([project_inclusion_probabilities(row) for row in raw])
    return blend_with_uniform(projected, blend_weight)


def _brier_by_draw(probabilities: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.mean((probabilities - actual) ** 2, axis=1)


def _log_loss(probabilities: np.ndarray, actual: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return float(np.mean(-(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped))))


def _calibration_error(probabilities: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    flat_p = probabilities.ravel()
    flat_y = actual.ravel()
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (flat_p >= lower) & (flat_p < upper if upper < 1 else flat_p <= upper)
        if mask.any():
            error += mask.mean() * abs(float(flat_p[mask].mean() - flat_y[mask].mean()))
    return float(error)


@dataclass(frozen=True, slots=True)
class MLBacktestResult:
    model: str
    outer_folds: int
    test_draws: int
    selected_parameters: tuple[dict[str, float | int], ...]
    final_parameters: dict[str, float | int]
    mean_brier: float
    uniform_brier: float
    mean_brier_delta: float
    delta_ci_low: float
    delta_ci_high: float
    permutation_p_value: float
    adjusted_p_value: float
    mean_log_loss: float
    uniform_log_loss: float
    calibration_error: float
    mean_top5_hits: float
    qualified: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _select_parameters(
    dataset: FeatureDataset,
    model_name: str,
    train_indices: np.ndarray,
    seed: int,
) -> dict[str, float | int]:
    train_dates = np.unique(dataset.dates[train_indices])
    validation_dates = train_dates[max(1, int(len(train_dates) * 0.8)) :]
    if len(validation_dates) < 1:
        return MODEL_PARAMETERS[model_name][0]
    split_date = validation_dates[0]
    inner_train = train_indices[dataset.dates[train_indices] < split_date]
    validation = train_indices[dataset.dates[train_indices] >= split_date]
    best: tuple[float, dict[str, float | int]] | None = None
    for model_parameters in MODEL_PARAMETERS[model_name]:
        raw_predicted = _fit_predict(
            model_name,
            model_parameters,
            dataset.x[inner_train],
            dataset.y[inner_train],
            dataset.x[validation],
            seed,
        )
        for blend_weight in UNIFORM_BLEND_WEIGHTS:
            predicted = blend_with_uniform(raw_predicted, blend_weight)
            score = float(_brier_by_draw(predicted, dataset.y[validation]).mean())
            parameters = {**model_parameters, "uniform_blend": blend_weight}
            if best is None or score < best[0]:
                best = (score, parameters)
    return dict(best[1])


def _confidence_and_permutation(
    deltas: np.ndarray, simulations: int, seed: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(simulations)]
    )
    observed = float(deltas.mean())
    extreme = 0
    for _ in range(simulations):
        signs = rng.choice((-1.0, 1.0), size=len(deltas))
        if float((deltas * signs).mean()) <= observed:
            extreme += 1
    return (
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
        (extreme + 1) / (simulations + 1),
    )


def _backtest_one_model(
    dataset: FeatureDataset,
    model_name: str,
    min_train: int,
    outer_folds: int,
    simulations: int,
    seed: int,
) -> MLBacktestResult:
    if dataset.draw_count <= min_train:
        raise ValueError("Pas assez de tirages pour le backtest ML")
    eligible_dates = np.unique(dataset.dates[min_train:])
    date_folds = [fold for fold in np.array_split(eligible_dates, outer_folds) if len(fold)]
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    selected: list[dict[str, float | int]] = []

    for fold_number, test_dates in enumerate(date_folds):
        train_indices = np.flatnonzero(dataset.dates < test_dates[0])
        test_indices = np.flatnonzero(np.isin(dataset.dates, test_dates))
        parameters = _select_parameters(dataset, model_name, train_indices, seed + fold_number)
        selected.append(parameters)
        predictions.append(
            _fit_predict(
                model_name,
                parameters,
                dataset.x[train_indices],
                dataset.y[train_indices],
                dataset.x[test_indices],
                seed + fold_number,
            )
        )
        actuals.append(dataset.y[test_indices])

    predicted = np.concatenate(predictions)
    actual = np.concatenate(actuals)
    baseline = np.full_like(predicted, 5 / 49)
    scores = _brier_by_draw(predicted, actual)
    baseline_scores = _brier_by_draw(baseline, actual)
    deltas = scores - baseline_scores
    ci_low, ci_high, p_value = _confidence_and_permutation(deltas, simulations, seed)
    hits = [
        len(set(np.argsort(row)[-5:]).intersection(np.flatnonzero(target)))
        for row, target in zip(predicted, actual, strict=True)
    ]
    final = _select_parameters(
        dataset, model_name, np.arange(dataset.draw_count), seed + outer_folds
    )
    return MLBacktestResult(
        model=model_name,
        outer_folds=len(date_folds),
        test_draws=len(actual),
        selected_parameters=tuple(selected),
        final_parameters=final,
        mean_brier=float(scores.mean()),
        uniform_brier=float(baseline_scores.mean()),
        mean_brier_delta=float(deltas.mean()),
        delta_ci_low=ci_low,
        delta_ci_high=ci_high,
        permutation_p_value=p_value,
        adjusted_p_value=1.0,
        mean_log_loss=_log_loss(predicted, actual),
        uniform_log_loss=_log_loss(baseline, actual),
        calibration_error=_calibration_error(predicted, actual),
        mean_top5_hits=sum(hits) / len(hits),
        qualified=False,
    )


def _holm_adjust(results: list[MLBacktestResult]) -> list[MLBacktestResult]:
    order = sorted(range(len(results)), key=lambda index: results[index].permutation_p_value)
    adjusted = [1.0] * len(results)
    running = 0.0
    total = len(results)
    for position, index in enumerate(order):
        value = min(1.0, (total - position) * results[index].permutation_p_value)
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


def nested_ml_backtest(
    draws: list[Draw],
    min_history: int = 50,
    min_train: int = 500,
    outer_folds: int = 3,
    simulations: int = 2_000,
    seed: int = 0,
    models: tuple[str, ...] = ("bayesian", "logistic", "gradient_boosting"),
) -> list[MLBacktestResult]:
    if outer_folds < 2 or simulations < 100:
        raise ValueError("Il faut au moins 2 folds et 100 simulations")
    unknown = set(models).difference(MODEL_PARAMETERS)
    if unknown:
        raise ValueError(f"Modeles inconnus: {sorted(unknown)}")
    dataset = build_feature_dataset(draws, min_history)
    results = [
        _backtest_one_model(
            dataset, model, min_train, outer_folds, simulations, seed + index * 10_000
        )
        for index, model in enumerate(models)
    ]
    return _holm_adjust(results)


def predict_next_draw(
    draws: list[Draw],
    results: list[MLBacktestResult],
    target_date: date,
    game: str = "loto",
    force: bool = False,
    seed: int = 0,
) -> dict[str, object]:
    qualified = [result for result in results if result.qualified]
    status = "qualified"
    if not qualified:
        if not force:
            return {
                "status": "abstention",
                "reason": "Aucun modele ne bat l'uniforme avec un intervalle a 95% et Holm < 5%.",
            }
        qualified = list(results)
        status = "forced_experimental"
    champion = min(qualified, key=lambda result: result.mean_brier_delta)
    dataset = build_feature_dataset(draws)
    target = next_feature_matrix(draws, target_date, game)
    predicted = _fit_predict(
        champion.model,
        champion.final_parameters,
        dataset.x,
        dataset.y,
        target[np.newaxis, :, :],
        seed,
    )[0]
    rng = random.Random(seed)
    if float(np.ptp(predicted)) < 1e-12:
        top = tuple(sorted(rng.sample(range(1, 50), 5)))
    else:
        top = tuple(sorted(int(index + 1) for index in np.argsort(predicted)[-5:]))
    return {
        "status": status,
        "model": champion.model,
        "parameters": champion.final_parameters,
        "numbers": top,
        "chance": rng.randint(1, 10),
        "marginal_probabilities": {
            number: float(predicted[number - 1]) for number in sorted(top)
        },
        "warning": "Une sortie forcee reste experimentale et ne constitue pas un avantage prouve.",
    }
