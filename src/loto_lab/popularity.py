from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import combinations, islice
from math import ceil, comb

import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from .domain import DEFAULT_RULES, Draw, Ticket
from .probability import rank_probabilities, total_outcomes

BASE_FEATURE_NAMES = (
    "numbers_above_31",
    "all_numbers_at_most_31",
    "consecutive_pairs",
    "lucky_numbers_7_13",
    "normalized_sum",
    "distance_from_central_sum",
)
INTERACTION_FEATURE_NAMES = BASE_FEATURE_NAMES + (
    "numbers_above_31_sum_interaction",
    "consecutive_sum_interaction",
    "numbers_above_31_squared",
    "consecutive_pairs_squared",
    "normalized_sum_squared",
)
NUMBER_EFFECT_FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(
    f"contains_number_{number}" for number in range(1, DEFAULT_RULES.main_pool + 1)
)
FEATURE_SETS = {
    "base": BASE_FEATURE_NAMES,
    "interactions": INTERACTION_FEATURE_NAMES,
    "number_effects": NUMBER_EFFECT_FEATURE_NAMES,
}
# Backward-compatible public alias used by v0.21/v0.22 evidence.
FEATURE_NAMES = BASE_FEATURE_NAMES
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
POPULARITY_TARGETS = ("jackpot", "main_combination")
POPULARITY_HYPOTHESES = len(POPULARITY_TARGETS) * len(FEATURE_SETS)


@dataclass(frozen=True, slots=True)
class PopularityObservation:
    draw: Draw
    jackpot_winners: int
    estimated_tickets: float
    exposure: float


@dataclass(frozen=True, slots=True)
class PopularityBacktestResult:
    game: str
    target: str
    observations: int
    test_observations: int
    outer_folds: int
    selected_alphas: tuple[float, ...]
    final_alpha: float
    mean_poisson_deviance: float
    baseline_poisson_deviance: float
    deviance_delta: float
    delta_ci_low: float
    delta_ci_high: float
    permutation_p_value: float
    adjusted_p_value: float
    temporal_block_size: int
    inference_method: str
    qualified: bool
    feature_set: str = "base"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PopularityPredictor:
    scaler: StandardScaler
    model: PoissonRegressor
    alpha: float
    observations: int
    baseline_multiplier: float
    validation: PopularityBacktestResult
    bootstrap_intercepts: np.ndarray
    bootstrap_coefficients: np.ndarray
    uncertainty_quantile: float
    feature_names: tuple[str, ...] = BASE_FEATURE_NAMES

    def multipliers(self, mains: np.ndarray, chances: np.ndarray) -> np.ndarray:
        features = _feature_matrix(mains, chances, self.feature_names)
        return self.model.predict(self.scaler.transform(features))

    def multiplier(self, ticket: Ticket) -> float:
        mains = np.asarray([ticket.main], dtype=int)
        chances = np.asarray([ticket.chance], dtype=int)
        return float(self.multipliers(mains, chances)[0])

    def conservative_multipliers(
        self, mains: np.ndarray, chances: np.ndarray
    ) -> np.ndarray:
        features = _feature_matrix(mains, chances, self.feature_names)
        log_relative = (
            self.bootstrap_intercepts[np.newaxis, :]
            + features @ self.bootstrap_coefficients.T
        )
        relative = np.exp(log_relative)
        return self.baseline_multiplier * np.quantile(
            relative, self.uncertainty_quantile, axis=1
        )


@dataclass(frozen=True, slots=True)
class ValueAwareSelection:
    ticket: Ticket
    top_probability_ticket: Ticket
    expected_main_hits: float
    top_expected_main_hits: float
    expected_hit_loss: float
    predicted_popularity_multiplier: float
    conservative_popularity_multiplier: float
    top_conservative_popularity_multiplier: float
    baseline_popularity_multiplier: float
    uncertainty_quantile: float
    bootstrap_models: int
    combinations_evaluated: int
    feasible_combinations: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ticket"] = {
            "main": self.ticket.main,
            "chance": self.ticket.chance,
        }
        payload["top_probability_ticket"] = {
            "main": self.top_probability_ticket.main,
            "chance": self.top_probability_ticket.chance,
        }
        return payload


def popularity_observations(
    draws: list[Draw], game: str = "loto", target: str = "jackpot"
) -> list[PopularityObservation]:
    if target not in POPULARITY_TARGETS:
        raise ValueError(f"Cible de popularite inconnue: {target}")
    rank_9_probability = {item.rank: item.probability for item in rank_probabilities()}[9]
    observations = []
    for draw in draws:
        if draw.game != game or draw.draw_date is None:
            continue
        prizes = {prize.rank: prize for prize in draw.prizes}
        rank_1 = prizes.get(1)
        rank_2 = prizes.get(2)
        rank_9 = prizes.get(9)
        if (
            rank_1 is None
            or rank_1.winners is None
            or (
                target == "main_combination"
                and (rank_2 is None or rank_2.winners is None)
            )
            or rank_9 is None
            or rank_9.winners is None
            or rank_9.winners <= 0
        ):
            continue
        estimated_tickets = rank_9.winners / rank_9_probability
        winners = rank_1.winners
        exposure = estimated_tickets / total_outcomes()
        if target == "main_combination":
            winners += rank_2.winners
            exposure = estimated_tickets / comb(
                DEFAULT_RULES.main_pool, DEFAULT_RULES.main_drawn
            )
        observations.append(
            PopularityObservation(
                draw,
                winners,
                estimated_tickets,
                exposure,
            )
        )
    return sorted(observations, key=lambda item: item.draw.draw_date)


def _feature_matrix(
    mains: np.ndarray,
    chances: np.ndarray,
    feature_names: tuple[str, ...] = BASE_FEATURE_NAMES,
) -> np.ndarray:
    if feature_names not in FEATURE_SETS.values():
        raise ValueError("Schema de variables de popularite inconnu")
    ordered = np.sort(np.asarray(mains, dtype=int), axis=1)
    gaps = np.diff(ordered, axis=1)
    base = np.column_stack(
        (
            np.sum(ordered > 31, axis=1),
            np.all(ordered <= 31, axis=1),
            np.sum(gaps == 1, axis=1),
            np.sum(np.isin(ordered, (7, 13)), axis=1),
            np.sum(ordered, axis=1) / 150,
            np.abs(np.sum(ordered, axis=1) - 125) / 100,
        )
    ).astype(float)
    if feature_names == BASE_FEATURE_NAMES:
        return base
    if feature_names == INTERACTION_FEATURE_NAMES:
        return np.column_stack(
            (
                base,
                base[:, 0] * base[:, 4],
                base[:, 2] * base[:, 4],
                base[:, 0] ** 2,
                base[:, 2] ** 2,
                base[:, 4] ** 2,
            )
        ).astype(float)
    number_effects = np.zeros((len(ordered), DEFAULT_RULES.main_pool), dtype=float)
    number_effects[np.arange(len(ordered))[:, np.newaxis], ordered - 1] = 1.0
    return np.column_stack((base, number_effects))


def _arrays(
    observations: list[PopularityObservation],
    feature_names: tuple[str, ...] = BASE_FEATURE_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates = np.asarray(
        [item.draw.draw_date.toordinal() for item in observations], dtype=int
    )
    mains = np.asarray([item.draw.main for item in observations], dtype=int)
    chances = np.asarray([item.draw.chance for item in observations], dtype=int)
    actual = np.asarray([item.jackpot_winners for item in observations], dtype=float)
    exposure = np.asarray([item.exposure for item in observations], dtype=float)
    return dates, _feature_matrix(mains, chances, feature_names), actual, exposure


def _fit(
    features: np.ndarray,
    actual: np.ndarray,
    exposure: np.ndarray,
    indices: np.ndarray,
    alpha: float,
) -> tuple[StandardScaler, PoissonRegressor]:
    scaler = StandardScaler().fit(features[indices])
    model = PoissonRegressor(alpha=alpha, max_iter=1_000)
    model.fit(
        scaler.transform(features[indices]),
        actual[indices] / exposure[indices],
        sample_weight=exposure[indices],
    )
    return scaler, model


def _predict(
    scaler: StandardScaler,
    model: PoissonRegressor,
    features: np.ndarray,
    exposure: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return model.predict(scaler.transform(features[indices])) * exposure[indices]


def _poisson_deviance(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    safe = np.maximum(predicted, 1e-12)
    terms = np.zeros_like(actual, dtype=float)
    positive = actual > 0
    terms[positive] = actual[positive] * np.log(actual[positive] / safe[positive])
    return 2 * (terms - (actual - safe))


def _select_alpha(
    dates: np.ndarray,
    features: np.ndarray,
    actual: np.ndarray,
    exposure: np.ndarray,
    train_indices: np.ndarray,
) -> float:
    train_dates = np.unique(dates[train_indices])
    validation_dates = train_dates[max(1, int(len(train_dates) * 0.8)) :]
    split_date = validation_dates[0]
    inner = train_indices[dates[train_indices] < split_date]
    validation = train_indices[dates[train_indices] >= split_date]
    candidates = []
    for alpha in ALPHAS:
        scaler, model = _fit(features, actual, exposure, inner, alpha)
        predicted = _predict(scaler, model, features, exposure, validation)
        candidates.append(
            (float(_poisson_deviance(actual[validation], predicted).mean()), alpha)
        )
    return min(candidates)[1]


def _moving_block_means(
    values: np.ndarray,
    simulations: int,
    rng: np.random.Generator,
    block_size: int,
) -> np.ndarray:
    blocks_needed = ceil(len(values) / block_size)
    means = []
    for _ in range(simulations):
        starts = rng.integers(0, len(values) - block_size + 1, size=blocks_needed)
        sample = np.concatenate(
            [values[start : start + block_size] for start in starts]
        )[: len(values)]
        means.append(float(sample.mean()))
    return np.asarray(means)


def popularity_backtest(
    draws: list[Draw],
    game: str = "loto",
    min_train: int = 500,
    outer_folds: int = 3,
    simulations: int = 2_000,
    block_size: int = 12,
    seed: int = 0,
    target: str = "jackpot",
    feature_set: str = "base",
) -> PopularityBacktestResult:
    if min_train < 100 or outer_folds < 2 or simulations < 100 or block_size < 1:
        raise ValueError(
            "Il faut min_train >= 100, 2 folds, 100 simulations et block_size >= 1"
        )
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Schema de variables inconnu: {feature_set}")
    feature_names = FEATURE_SETS[feature_set]
    observations = popularity_observations(draws, game, target)
    if len(observations) <= min_train:
        raise ValueError("Historique insuffisant pour le modele de popularite")
    dates, features, actual, exposure = _arrays(observations, feature_names)
    if actual.sum() <= 0:
        raise ValueError("Aucun gagnant pour ajuster la popularite")
    eligible_dates = np.unique(dates[min_train:])
    folds = [fold for fold in np.array_split(eligible_dates, outer_folds) if len(fold)]
    model_scores = []
    baseline_scores = []
    selected_alphas = []
    for test_dates in folds:
        train = np.flatnonzero(dates < test_dates[0])
        test = np.flatnonzero(np.isin(dates, test_dates))
        alpha = _select_alpha(dates, features, actual, exposure, train)
        selected_alphas.append(alpha)
        scaler, model = _fit(features, actual, exposure, train, alpha)
        predicted = _predict(scaler, model, features, exposure, test)
        baseline_multiplier = actual[train].sum() / exposure[train].sum()
        baseline = baseline_multiplier * exposure[test]
        model_scores.extend(_poisson_deviance(actual[test], predicted))
        baseline_scores.extend(_poisson_deviance(actual[test], baseline))

    model_values = np.asarray(model_scores)
    baseline_values = np.asarray(baseline_scores)
    deltas = model_values - baseline_values
    effective_block = min(block_size, len(deltas))
    rng = np.random.default_rng(seed)
    bootstrap = _moving_block_means(deltas, simulations, rng, effective_block)
    blocks_needed = ceil(len(deltas) / effective_block)
    extreme = 0
    for _ in range(simulations):
        signs = np.repeat(
            rng.choice((-1.0, 1.0), size=blocks_needed), effective_block
        )[: len(deltas)]
        if float((deltas * signs).mean()) <= float(deltas.mean()):
            extreme += 1
    p_value = (extreme + 1) / (simulations + 1)
    adjusted_p_value = min(1.0, p_value * POPULARITY_HYPOTHESES)
    low = float(np.quantile(bootstrap, 0.025))
    high = float(np.quantile(bootstrap, 0.975))
    final_alpha = _select_alpha(
        dates, features, actual, exposure, np.arange(len(observations))
    )
    return PopularityBacktestResult(
        game=game,
        target=target,
        observations=len(observations),
        test_observations=len(deltas),
        outer_folds=len(folds),
        selected_alphas=tuple(selected_alphas),
        final_alpha=final_alpha,
        mean_poisson_deviance=float(model_values.mean()),
        baseline_poisson_deviance=float(baseline_values.mean()),
        deviance_delta=float(deltas.mean()),
        delta_ci_low=low,
        delta_ci_high=high,
        permutation_p_value=p_value,
        adjusted_p_value=adjusted_p_value,
        temporal_block_size=effective_block,
        inference_method="moving_block_bootstrap_and_block_sign_permutation",
        qualified=high < 0 and adjusted_p_value < 0.05,
        feature_set=feature_set,
    )


def fit_popularity_predictor(
    draws: list[Draw],
    result: PopularityBacktestResult,
    bootstrap_models: int = 100,
    uncertainty_quantile: float = 0.9,
    seed: int = 0,
) -> PopularityPredictor:
    if bootstrap_models < 20 or not 0.5 <= uncertainty_quantile < 1:
        raise ValueError(
            "Il faut au moins 20 modeles bootstrap et un quantile dans [0,5; 1["
        )
    observations = popularity_observations(draws, result.game, result.target)
    feature_names = FEATURE_SETS[result.feature_set]
    _, features, actual, exposure = _arrays(observations, feature_names)
    indices = np.arange(len(observations))
    scaler, model = _fit(
        features, actual, exposure, indices, result.final_alpha
    )
    block_size = min(result.temporal_block_size, len(observations))
    blocks_needed = ceil(len(observations) / block_size)
    rng = np.random.default_rng(seed)
    bootstrap_intercepts = []
    bootstrap_coefficients = []
    while len(bootstrap_intercepts) < bootstrap_models:
        starts = rng.integers(
            0, len(observations) - block_size + 1, size=blocks_needed
        )
        sample = np.concatenate(
            [np.arange(start, start + block_size) for start in starts]
        )[: len(observations)]
        if actual[sample].sum() <= 0:
            continue
        sample_scaler, sample_model = _fit(
            features, actual, exposure, sample, result.final_alpha
        )
        sample_baseline = actual[sample].sum() / exposure[sample].sum()
        raw_coefficients = sample_model.coef_ / sample_scaler.scale_
        raw_intercept = (
            sample_model.intercept_
            - np.dot(
                sample_model.coef_, sample_scaler.mean_ / sample_scaler.scale_
            )
            - np.log(sample_baseline)
        )
        bootstrap_intercepts.append(float(raw_intercept))
        bootstrap_coefficients.append(raw_coefficients)
    return PopularityPredictor(
        scaler=scaler,
        model=model,
        alpha=result.final_alpha,
        observations=len(observations),
        baseline_multiplier=float(actual.sum() / exposure.sum()),
        validation=result,
        bootstrap_intercepts=np.asarray(bootstrap_intercepts),
        bootstrap_coefficients=np.asarray(bootstrap_coefficients),
        uncertainty_quantile=uncertainty_quantile,
        feature_names=feature_names,
    )


def _combination_batches(batch_size: int) -> Iterator[np.ndarray]:
    iterator = combinations(range(1, DEFAULT_RULES.main_pool + 1), 5)
    while batch := list(islice(iterator, batch_size)):
        yield np.asarray(batch, dtype=int)


def optimize_value_aware_ticket(
    probabilities: np.ndarray,
    predictor: PopularityPredictor,
    max_expected_hit_loss: float = 0.005,
    batch_size: int = 50_000,
    chance: int = 1,
) -> ValueAwareSelection:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (DEFAULT_RULES.main_pool,) or not np.all(np.isfinite(values)):
        raise ValueError("Il faut 49 probabilites marginales finies")
    if np.any(values < 0) or max_expected_hit_loss < 0 or batch_size < 1:
        raise ValueError("Probabilites, perte maximale et batch_size doivent etre positifs")
    DEFAULT_RULES.validate_chance(chance)
    top_indices = np.argsort(values, kind="stable")[-DEFAULT_RULES.main_drawn :]
    top_main = tuple(sorted(int(index + 1) for index in top_indices))
    top_score = float(values[top_indices].sum())
    threshold = top_score - max_expected_hit_loss
    best: tuple[float, float, tuple[int, ...], int] | None = None
    evaluated = 0
    feasible = 0

    for mains in _combination_batches(batch_size):
        evaluated += len(mains)
        scores = values[mains - 1].sum(axis=1)
        mask = scores >= threshold - 1e-12
        if not mask.any():
            continue
        candidates = mains[mask]
        candidate_scores = scores[mask]
        feasible += len(candidates)
        candidate_chances = np.full(len(candidates), chance, dtype=int)
        conservative = predictor.conservative_multipliers(
            candidates, candidate_chances
        )
        order = np.lexsort(
            (
                candidate_chances,
                candidates[:, 4],
                candidates[:, 3],
                candidates[:, 2],
                candidates[:, 1],
                candidates[:, 0],
                -candidate_scores,
                conservative,
            )
        )
        position = int(order[0])
        key = (
            float(conservative[position]),
            -float(candidate_scores[position]),
            tuple(int(value) for value in candidates[position]),
            int(candidate_chances[position]),
        )
        if best is None or key < best:
            best = key

    if best is None:
        raise RuntimeError("Aucune combinaison admissible trouvee")
    selected_score = -best[1]
    ticket = Ticket(best[2], best[3])
    top_ticket = Ticket(top_main, chance)
    return ValueAwareSelection(
        ticket=ticket,
        top_probability_ticket=top_ticket,
        expected_main_hits=selected_score,
        top_expected_main_hits=top_score,
        expected_hit_loss=top_score - selected_score,
        predicted_popularity_multiplier=predictor.multiplier(ticket),
        conservative_popularity_multiplier=best[0],
        top_conservative_popularity_multiplier=float(
            predictor.conservative_multipliers(
                np.asarray([top_ticket.main]), np.asarray([top_ticket.chance])
            )[0]
        ),
        baseline_popularity_multiplier=predictor.baseline_multiplier,
        uncertainty_quantile=predictor.uncertainty_quantile,
        bootstrap_models=len(predictor.bootstrap_intercepts),
        combinations_evaluated=evaluated,
        feasible_combinations=feasible,
    )
