from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import combinations, islice
from math import ceil

import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from .domain import DEFAULT_RULES, Draw, Ticket
from .probability import rank_probabilities, total_outcomes

FEATURE_NAMES = (
    "numbers_above_31",
    "all_numbers_at_most_31",
    "consecutive_pairs",
    "lucky_numbers_7_13",
    "normalized_sum",
    "distance_from_central_sum",
)
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True, slots=True)
class PopularityObservation:
    draw: Draw
    jackpot_winners: int
    estimated_tickets: float
    exposure: float


@dataclass(frozen=True, slots=True)
class PopularityBacktestResult:
    game: str
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
    temporal_block_size: int
    inference_method: str
    qualified: bool

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

    def multipliers(self, mains: np.ndarray, chances: np.ndarray) -> np.ndarray:
        features = _feature_matrix(mains, chances)
        return self.model.predict(self.scaler.transform(features))

    def multiplier(self, ticket: Ticket) -> float:
        mains = np.asarray([ticket.main], dtype=int)
        chances = np.asarray([ticket.chance], dtype=int)
        return float(self.multipliers(mains, chances)[0])


@dataclass(frozen=True, slots=True)
class ValueAwareSelection:
    ticket: Ticket
    top_probability_ticket: Ticket
    expected_main_hits: float
    top_expected_main_hits: float
    expected_hit_loss: float
    predicted_popularity_multiplier: float
    baseline_popularity_multiplier: float
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
    draws: list[Draw], game: str = "loto"
) -> list[PopularityObservation]:
    rank_9_probability = {item.rank: item.probability for item in rank_probabilities()}[9]
    observations = []
    for draw in draws:
        if draw.game != game or draw.draw_date is None:
            continue
        prizes = {prize.rank: prize for prize in draw.prizes}
        rank_1 = prizes.get(1)
        rank_9 = prizes.get(9)
        if (
            rank_1 is None
            or rank_1.winners is None
            or rank_9 is None
            or rank_9.winners is None
            or rank_9.winners <= 0
        ):
            continue
        estimated_tickets = rank_9.winners / rank_9_probability
        observations.append(
            PopularityObservation(
                draw,
                rank_1.winners,
                estimated_tickets,
                estimated_tickets / total_outcomes(),
            )
        )
    return sorted(observations, key=lambda item: item.draw.draw_date)


def _feature_matrix(mains: np.ndarray, chances: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(mains, dtype=int), axis=1)
    gaps = np.diff(ordered, axis=1)
    return np.column_stack(
        (
            np.sum(ordered > 31, axis=1),
            np.all(ordered <= 31, axis=1),
            np.sum(gaps == 1, axis=1),
            np.sum(np.isin(ordered, (7, 13)), axis=1),
            np.sum(ordered, axis=1) / 150,
            np.abs(np.sum(ordered, axis=1) - 125) / 100,
        )
    ).astype(float)


def _arrays(
    observations: list[PopularityObservation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates = np.asarray(
        [item.draw.draw_date.toordinal() for item in observations], dtype=int
    )
    mains = np.asarray([item.draw.main for item in observations], dtype=int)
    chances = np.asarray([item.draw.chance for item in observations], dtype=int)
    actual = np.asarray([item.jackpot_winners for item in observations], dtype=float)
    exposure = np.asarray([item.exposure for item in observations], dtype=float)
    return dates, _feature_matrix(mains, chances), actual, exposure


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
) -> PopularityBacktestResult:
    if min_train < 100 or outer_folds < 2 or simulations < 100 or block_size < 1:
        raise ValueError(
            "Il faut min_train >= 100, 2 folds, 100 simulations et block_size >= 1"
        )
    observations = popularity_observations(draws, game)
    if len(observations) <= min_train:
        raise ValueError("Historique insuffisant pour le modele de popularite")
    dates, features, actual, exposure = _arrays(observations)
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
    low = float(np.quantile(bootstrap, 0.025))
    high = float(np.quantile(bootstrap, 0.975))
    final_alpha = _select_alpha(
        dates, features, actual, exposure, np.arange(len(observations))
    )
    return PopularityBacktestResult(
        game=game,
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
        temporal_block_size=effective_block,
        inference_method="moving_block_bootstrap_and_block_sign_permutation",
        qualified=high < 0 and p_value < 0.05,
    )


def fit_popularity_predictor(
    draws: list[Draw], result: PopularityBacktestResult
) -> PopularityPredictor:
    observations = popularity_observations(draws, result.game)
    _, features, actual, exposure = _arrays(observations)
    indices = np.arange(len(observations))
    scaler, model = _fit(
        features, actual, exposure, indices, result.final_alpha
    )
    return PopularityPredictor(
        scaler=scaler,
        model=model,
        alpha=result.final_alpha,
        observations=len(observations),
        baseline_multiplier=float(actual.sum() / exposure.sum()),
        validation=result,
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
        minimums = predictor.multipliers(candidates, candidate_chances)
        order = np.lexsort(
            (
                candidate_chances,
                candidates[:, 4],
                candidates[:, 3],
                candidates[:, 2],
                candidates[:, 1],
                candidates[:, 0],
                -candidate_scores,
                minimums,
            )
        )
        position = int(order[0])
        key = (
            float(minimums[position]),
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
    return ValueAwareSelection(
        ticket=ticket,
        top_probability_ticket=Ticket(top_main, chance),
        expected_main_hits=selected_score,
        top_expected_main_hits=top_score,
        expected_hit_loss=top_score - selected_score,
        predicted_popularity_multiplier=best[0],
        baseline_popularity_multiplier=predictor.baseline_multiplier,
        combinations_evaluated=evaluated,
        feasible_combinations=feasible,
    )
