from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import numpy as np

from .domain import DEFAULT_RULES, Draw

CHANCE_PRIORS = (10.0, 30.0, 100.0, 300.0, 1_000.0)
CHANCE_INFERENCE_SIMULATIONS = 2_000
CHANCE_BLOCK_SIZE = 12


@dataclass(frozen=True, slots=True)
class ChanceObservation:
    draw: Draw
    correct_chance_winners: int
    main_combination_winners: int


@dataclass(frozen=True, slots=True)
class ChanceBacktestResult:
    game: str
    observations: int
    test_observations: int
    outer_folds: int
    selected_priors: tuple[float, ...]
    final_prior: float
    mean_binomial_deviance: float
    baseline_binomial_deviance: float
    deviance_delta: float
    delta_ci_low: float
    delta_ci_high: float
    permutation_p_value: float
    chance_factors: tuple[float, ...]
    first_half_factors: tuple[float, ...]
    last_half_factors: tuple[float, ...]
    qualified: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def chance_observations(
    draws: list[Draw], game: str = "loto"
) -> list[ChanceObservation]:
    observations: list[ChanceObservation] = []
    for draw in draws:
        if draw.game != game or draw.draw_date is None:
            continue
        prizes = {prize.rank: prize for prize in draw.prizes}
        rank_1 = prizes.get(1)
        rank_2 = prizes.get(2)
        if (
            rank_1 is None
            or rank_1.winners is None
            or rank_2 is None
            or rank_2.winners is None
            or rank_1.winners < 0
            or rank_2.winners < 0
            or rank_1.winners + rank_2.winners <= 0
        ):
            continue
        observations.append(
            ChanceObservation(
                draw,
                rank_1.winners,
                rank_1.winners + rank_2.winners,
            )
        )
    return sorted(observations, key=lambda item: item.draw.draw_date)


def _binomial_deviance(
    successes: np.ndarray, totals: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    safe = np.clip(probabilities, 1e-12, 1 - 1e-12)
    failures = totals - successes
    terms = np.zeros_like(safe, dtype=float)
    positive_successes = successes > 0
    positive_failures = failures > 0
    terms[positive_successes] += successes[positive_successes] * np.log(
        successes[positive_successes] / (totals[positive_successes] * safe[positive_successes])
    )
    terms[positive_failures] += failures[positive_failures] * np.log(
        failures[positive_failures]
        / (totals[positive_failures] * (1 - safe[positive_failures]))
    )
    return 2 * terms


def _posterior(
    observations: list[ChanceObservation], indices: np.ndarray, prior: float
) -> np.ndarray:
    successes = np.full(DEFAULT_RULES.chance_pool, prior / DEFAULT_RULES.chance_pool)
    totals = np.full(DEFAULT_RULES.chance_pool, prior)
    for index in indices:
        item = observations[int(index)]
        chance = item.draw.chance - 1
        successes[chance] += item.correct_chance_winners
        totals[chance] += item.main_combination_winners
    return successes / totals


def _select_prior(observations: list[ChanceObservation], train: np.ndarray) -> float:
    dates = np.asarray([item.draw.draw_date.toordinal() for item in observations])
    train_dates = np.unique(dates[train])
    if len(train_dates) < 2:
        return CHANCE_PRIORS[0]
    split = train_dates[min(max(1, int(len(train_dates) * 0.8)), len(train_dates) - 1)]
    inner = train[dates[train] < split]
    validation = train[dates[train] >= split]
    if not len(inner) or not len(validation):
        return CHANCE_PRIORS[0]
    candidates = []
    for prior in CHANCE_PRIORS:
        factors = _posterior(observations, inner, prior)
        successes = np.asarray([observations[i].correct_chance_winners for i in validation])
        totals = np.asarray([observations[i].main_combination_winners for i in validation])
        probabilities = np.asarray([factors[observations[i].draw.chance - 1] for i in validation])
        candidates.append(
            (float(_binomial_deviance(successes, totals, probabilities).mean()), prior)
        )
    return min(candidates)[1]


def _moving_block_means(
    values: np.ndarray, simulations: int, rng: np.random.Generator
) -> np.ndarray:
    block = min(CHANCE_BLOCK_SIZE, len(values))
    blocks = ceil(len(values) / block)
    means = []
    for _ in range(simulations):
        starts = rng.integers(0, len(values) - block + 1, size=blocks)
        sample = np.concatenate([values[start : start + block] for start in starts])[: len(values)]
        means.append(float(sample.mean()))
    return np.asarray(means)


def chance_popularity_backtest(
    draws: list[Draw],
    game: str = "loto",
    min_train: int = 500,
    outer_folds: int = 3,
    simulations: int = CHANCE_INFERENCE_SIMULATIONS,
    seed: int = 0,
) -> ChanceBacktestResult:
    if min_train < 100 or outer_folds < 2 or simulations < 100:
        raise ValueError("Il faut min_train >= 100, 2 folds et 100 simulations")
    observations = chance_observations(draws, game)
    if len(observations) <= min_train:
        raise ValueError("Historique insuffisant pour la popularite Chance")
    dates = np.asarray([item.draw.draw_date.toordinal() for item in observations])
    eligible = np.unique(dates[min_train:])
    folds = [fold for fold in np.array_split(eligible, outer_folds) if len(fold)]
    model_scores: list[float] = []
    baseline_scores: list[float] = []
    selected: list[float] = []
    for test_dates in folds:
        train = np.flatnonzero(dates < test_dates[0])
        test = np.flatnonzero(np.isin(dates, test_dates))
        prior = _select_prior(observations, train)
        selected.append(prior)
        factors = _posterior(observations, train, prior)
        successes = np.asarray([observations[i].correct_chance_winners for i in test])
        totals = np.asarray([observations[i].main_combination_winners for i in test])
        probabilities = np.asarray([factors[observations[i].draw.chance - 1] for i in test])
        model_scores.extend(_binomial_deviance(successes, totals, probabilities))
        baseline_scores.extend(_binomial_deviance(successes, totals, np.full(len(test), 0.1)))
    deltas = np.asarray(model_scores) - np.asarray(baseline_scores)
    rng = np.random.default_rng(seed)
    bootstrap = _moving_block_means(deltas, simulations, rng)
    block = min(CHANCE_BLOCK_SIZE, len(deltas))
    blocks = ceil(len(deltas) / block)
    observed = float(deltas.mean())
    extreme = 0
    for _ in range(simulations):
        signs = np.repeat(rng.choice((-1.0, 1.0), size=blocks), block)[: len(deltas)]
        extreme += float((deltas * signs).mean()) <= observed
    final_prior = _select_prior(observations, np.arange(len(observations)))
    factors = _posterior(observations, np.arange(len(observations)), final_prior)
    midpoint = len(observations) // 2
    first = _posterior(observations, np.arange(midpoint), final_prior) / 0.1
    last = _posterior(observations, np.arange(midpoint, len(observations)), final_prior) / 0.1
    p_value = (extreme + 1) / (simulations + 1)
    return ChanceBacktestResult(
        game=game,
        observations=len(observations),
        test_observations=len(deltas),
        outer_folds=len(folds),
        selected_priors=tuple(selected),
        final_prior=final_prior,
        mean_binomial_deviance=float(np.mean(model_scores)),
        baseline_binomial_deviance=float(np.mean(baseline_scores)),
        deviance_delta=float(observed),
        delta_ci_low=float(np.quantile(bootstrap, 0.025)),
        delta_ci_high=float(np.quantile(bootstrap, 0.975)),
        permutation_p_value=p_value,
        chance_factors=tuple(float(value / 0.1) for value in factors),
        first_half_factors=tuple(float(value) for value in first),
        last_half_factors=tuple(float(value) for value in last),
        qualified=bool(np.quantile(bootstrap, 0.975) < 0 and p_value < 0.05),
    )
