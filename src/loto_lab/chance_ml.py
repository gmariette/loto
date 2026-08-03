from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import ceil, exp, lgamma, log

import numpy as np

from .domain import DEFAULT_RULES, Draw

CHANCE_PARAMETER_GRID = tuple(
    {
        "window": window,
        "prior_strength": prior,
        "transition_weight": transition_weight,
        "weekday_weight": weekday_weight,
    }
    for window in (25, 100, 400, 0)
    for prior in (20.0, 100.0, 500.0)
    for transition_weight in (0.0, 0.25)
    for weekday_weight in (0.0, 0.25)
)
DEFAULT_CHANCE_PARAMETERS = {
    "window": 400,
    "prior_strength": 100.0,
    "transition_weight": 0.0,
    "weekday_weight": 0.0,
}
# Brier, log-loss and Top-1 are tested jointly against the uniform reference.
CHANCE_HYPOTHESES = 3
TIE_TOLERANCE = 1e-12


def uniform_probability(pool: int = DEFAULT_RULES.chance_pool) -> float:
    """Probabilite de reference d'un numero Chance sous l'hypothese uniforme."""
    return 1.0 / pool


def uniform_brier(pool: int = DEFAULT_RULES.chance_pool) -> float:
    """Brier multiclasse (moyenne sur les classes) d'une prevision uniforme."""
    return (pool - 1) / pool**2


def uniform_log_loss(pool: int = DEFAULT_RULES.chance_pool) -> float:
    """Log-loss multiclasse d'une prevision uniforme."""
    return log(pool)


@dataclass(frozen=True, slots=True)
class ChanceMLBacktestResult:
    game: str
    observations: int
    test_observations: int
    outer_folds: int
    selected_parameters: tuple[dict[str, float | int], ...]
    final_parameters: dict[str, float | int]
    mean_brier: float
    uniform_brier: float
    mean_brier_delta: float
    delta_ci_low: float
    delta_ci_high: float
    permutation_p_value: float
    adjusted_brier_p_value: float
    mean_log_loss: float
    uniform_log_loss: float
    mean_log_loss_delta: float
    log_loss_ci_low: float
    log_loss_ci_high: float
    log_loss_p_value: float
    adjusted_log_loss_p_value: float
    calibration_error: float
    uniform_calibration_error: float
    calibration_bins: tuple[dict[str, float], ...]
    top1_accuracy: float
    uniform_top1_accuracy: float
    top1_uplift: float
    top1_ci_low: float
    top1_ci_high: float
    top1_p_value: float
    adjusted_top1_p_value: float
    top1_test_method: str
    top1_tied_draws: int
    hypotheses: int
    qualification_basis: str
    qualified: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ordered_game_draws(draws: list[Draw], game: str) -> list[Draw]:
    selected = [
        draw for draw in draws if draw.game == game and draw.draw_date is not None
    ]
    if not selected:
        raise ValueError(f"Aucun tirage date disponible pour {game}")
    return sorted(selected, key=lambda draw: (draw.draw_date, draw.main, draw.chance))


def _smoothed_distribution(
    counts: np.ndarray, observations: int, prior_strength: float
) -> np.ndarray:
    return (counts + prior_strength / DEFAULT_RULES.chance_pool) / (
        observations + prior_strength
    )


def _distribution_from_state(
    rolling_counts: np.ndarray,
    rolling_observations: int,
    transitions: np.ndarray,
    previous: int | None,
    weekday_counts: np.ndarray,
    weekday_observations: np.ndarray,
    target_weekday: int,
    parameters: dict[str, float | int],
) -> np.ndarray:
    prior = float(parameters["prior_strength"])
    base = _smoothed_distribution(rolling_counts, rolling_observations, prior)
    transition_weight = float(parameters["transition_weight"])
    weekday_weight = float(parameters["weekday_weight"])
    prediction = (1 - transition_weight - weekday_weight) * base
    if transition_weight:
        transition_counts = (
            transitions[previous]
            if previous is not None
            else np.zeros(DEFAULT_RULES.chance_pool)
        )
        prediction += transition_weight * _smoothed_distribution(
            transition_counts, int(transition_counts.sum()), prior
        )
    if weekday_weight:
        prediction += weekday_weight * _smoothed_distribution(
            weekday_counts[target_weekday],
            int(weekday_observations[target_weekday]),
            prior,
        )
    return prediction / prediction.sum()


def _prediction_series(
    chances: np.ndarray,
    weekdays: np.ndarray,
    parameters: dict[str, float | int],
) -> np.ndarray:
    pool = DEFAULT_RULES.chance_pool
    window = int(parameters["window"])
    rolling_counts = np.zeros(pool, dtype=float)
    transitions = np.zeros((pool, pool), dtype=float)
    weekday_counts = np.zeros((7, pool), dtype=float)
    weekday_observations = np.zeros(7, dtype=int)
    predictions = np.empty((len(chances), pool), dtype=float)
    previous: int | None = None
    for index, (chance, weekday) in enumerate(zip(chances, weekdays, strict=True)):
        predictions[index] = _distribution_from_state(
            rolling_counts,
            min(index, window) if window else index,
            transitions,
            previous,
            weekday_counts,
            weekday_observations,
            int(weekday),
            parameters,
        )
        if window and index >= window:
            rolling_counts[chances[index - window]] -= 1
        rolling_counts[chance] += 1
        if previous is not None:
            transitions[previous, chance] += 1
        weekday_counts[weekday, chance] += 1
        weekday_observations[weekday] += 1
        previous = int(chance)
    return predictions


def _predict_after_history(
    chances: np.ndarray,
    weekdays: np.ndarray,
    target_weekday: int,
    parameters: dict[str, float | int],
) -> np.ndarray:
    pool = DEFAULT_RULES.chance_pool
    window = int(parameters["window"])
    selected = chances[-window:] if window else chances
    rolling_counts = np.bincount(selected, minlength=pool).astype(float)
    transitions = np.zeros((pool, pool), dtype=float)
    if len(chances) > 1:
        np.add.at(transitions, (chances[:-1], chances[1:]), 1)
    weekday_counts = np.zeros((7, pool), dtype=float)
    np.add.at(weekday_counts, (weekdays, chances), 1)
    weekday_observations = np.bincount(weekdays, minlength=7)
    return _distribution_from_state(
        rolling_counts,
        len(selected),
        transitions,
        int(chances[-1]),
        weekday_counts,
        weekday_observations,
        target_weekday,
        parameters,
    )


def _one_hot(probabilities: np.ndarray, chances: np.ndarray) -> np.ndarray:
    actual = np.zeros_like(probabilities)
    actual[np.arange(len(chances)), chances] = 1.0
    return actual


def _brier_by_draw(probabilities: np.ndarray, chances: np.ndarray) -> np.ndarray:
    return np.mean((probabilities - _one_hot(probabilities, chances)) ** 2, axis=1)


def _log_loss_by_draw(probabilities: np.ndarray, chances: np.ndarray) -> np.ndarray:
    selected = probabilities[np.arange(len(chances)), chances]
    return -np.log(np.clip(selected, 1e-15, 1.0))


def argmax_with_ties(probabilities: np.ndarray) -> tuple[int, np.ndarray]:
    """Renvoie le numero Chance maximal et l'ensemble complet des ex aequo."""
    tied = np.flatnonzero(probabilities >= probabilities.max() - TIE_TOLERANCE)
    return int(tied[0]) + 1, tied + 1


def _expected_top1_hits(
    probabilities: np.ndarray, chances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Esperance de Top-1 par tirage en resolvant les ex aequo uniformement.

    Un `argmax` brut departagerait arbitrairement les ex aequo en faveur du plus
    petit numero; l'esperance est deterministe et sans biais.
    """
    maxima = probabilities.max(axis=1, keepdims=True)
    tied = probabilities >= maxima - TIE_TOLERANCE
    tie_sizes = tied.sum(axis=1)
    selected = tied[np.arange(len(chances)), chances]
    return selected / tie_sizes, tie_sizes


def _calibration(
    probabilities: np.ndarray, chances: np.ndarray, bins: int = 10
) -> tuple[float, tuple[dict[str, float], ...]]:
    """Erreur de calibration esperee sur des bacs de quantiles.

    Des bacs fixes sur [0, 1] placeraient toutes les probabilites (~0,10) dans le
    meme bac et ne mesureraient rien.
    """
    flat_p = probabilities.ravel()
    flat_y = _one_hot(probabilities, chances).ravel()
    quantiles = np.quantile(flat_p, np.linspace(0, 1, bins + 1))
    edges = np.unique(quantiles)
    if len(edges) < 2:
        return 0.0, ()
    assignment = np.clip(np.searchsorted(edges, flat_p, side="left") - 1, 0, len(edges) - 2)
    error = 0.0
    summary: list[dict[str, float]] = []
    for index in range(len(edges) - 1):
        mask = assignment == index
        if not mask.any():
            continue
        weight = float(mask.mean())
        mean_p = float(flat_p[mask].mean())
        mean_y = float(flat_y[mask].mean())
        error += weight * abs(mean_p - mean_y)
        summary.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "weight": weight,
                "mean_predicted": mean_p,
                "observed_rate": mean_y,
            }
        )
    return float(error), tuple(summary)


def _select_parameters(
    dates: np.ndarray,
    chances: np.ndarray,
    prediction_cache: dict[tuple[object, ...], np.ndarray],
    train: np.ndarray,
) -> dict[str, float | int]:
    """Choisit les hyperparametres sur la fin du passe d'entrainement seulement."""
    train_dates = np.unique(dates[train])
    if len(train_dates) < 2:
        return dict(DEFAULT_CHANCE_PARAMETERS)
    split_index = min(max(1, int(len(train_dates) * 0.8)), len(train_dates) - 1)
    split = train_dates[split_index]
    validation = train[dates[train] >= split]
    if not len(validation):
        return dict(DEFAULT_CHANCE_PARAMETERS)
    candidates = []
    for parameters in CHANCE_PARAMETER_GRID:
        key = tuple(parameters.values())
        probabilities = prediction_cache[key][validation]
        candidates.append(
            (
                # Objectif de selection identique a la metrique publiee: log-loss
                # puis Brier, tous deux des regles de score propres.
                float(_log_loss_by_draw(probabilities, chances[validation]).mean()),
                float(_brier_by_draw(probabilities, chances[validation]).mean()),
                int(parameters["window"]) == 0,
                -float(parameters["prior_strength"]),
                float(parameters["transition_weight"])
                + float(parameters["weekday_weight"]),
                parameters,
            )
        )
    return dict(min(candidates, key=lambda item: item[:-1])[-1])


def _binomial_upper_tail(draws: int, hits: int, probability: float) -> float:
    if not 0 <= hits <= draws or not 0 < probability < 1:
        raise ValueError("Parametres binomiaux invalides")
    terms = [
        exp(
            lgamma(draws + 1)
            - lgamma(count + 1)
            - lgamma(draws - count + 1)
            + count * log(probability)
            + (draws - count) * log(1 - probability)
        )
        for count in range(hits, draws + 1)
    ]
    return min(1.0, float(sum(terms)))


def _top1_p_value(
    hits: np.ndarray,
    tie_sizes: np.ndarray,
    pool: int,
    simulations: int,
    rng: np.random.Generator,
) -> tuple[float, str]:
    """p-value unilaterale du taux de Top-1 contre un Chance uniforme IID.

    Sans ex aequo, la loi exacte des succes est binomiale. Avec ex aequo, la
    statistique devient fractionnaire: on simule alors la loi nulle exacte induite
    par la structure d'ex aequo observee.
    """
    if np.all(tie_sizes == 1):
        return (
            _binomial_upper_tail(len(hits), int(round(hits.sum())), 1.0 / pool),
            "exact_binomial",
        )
    probabilities = tie_sizes / pool
    contributions = 1.0 / tie_sizes
    observed = float(hits.mean())
    null_means = (
        (rng.random((simulations, len(hits))) < probabilities) * contributions
    ).mean(axis=1)
    extreme = int(np.count_nonzero(null_means >= observed))
    return (extreme + 1) / (simulations + 1), "monte_carlo_tie_aware"


def _moving_block_means(
    values: np.ndarray,
    simulations: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    effective_block = min(block_size, len(values))
    blocks_needed = ceil(len(values) / effective_block)
    means = []
    for _ in range(simulations):
        starts = rng.integers(
            0, len(values) - effective_block + 1, size=blocks_needed
        )
        sample = np.concatenate(
            [values[start : start + effective_block] for start in starts]
        )[: len(values)]
        means.append(float(sample.mean()))
    return np.asarray(means)


def _block_sign_p_value(
    deltas: np.ndarray, simulations: int, block_size: int, rng: np.random.Generator
) -> float:
    effective_block = min(block_size, len(deltas))
    blocks_needed = ceil(len(deltas) / effective_block)
    observed = float(deltas.mean())
    extreme = 0
    for _ in range(simulations):
        signs = np.repeat(
            rng.choice((-1.0, 1.0), size=blocks_needed), effective_block
        )[: len(deltas)]
        extreme += float((deltas * signs).mean()) <= observed
    return (extreme + 1) / (simulations + 1)


def chance_ml_backtest(
    draws: list[Draw],
    game: str = "loto",
    min_train: int = 500,
    outer_folds: int = 3,
    simulations: int = 2_000,
    block_size: int = 12,
    seed: int = 0,
) -> ChanceMLBacktestResult:
    if min_train < 100 or outer_folds < 2 or simulations < 100 or block_size < 1:
        raise ValueError(
            "Il faut min_train >= 100, 2 folds, 100 simulations et block_size >= 1"
        )
    pool = DEFAULT_RULES.chance_pool
    ordered = _ordered_game_draws(draws, game)
    if len(ordered) <= min_train:
        raise ValueError("Historique insuffisant pour le modele Chance")
    dates = np.asarray([draw.draw_date.toordinal() for draw in ordered], dtype=int)
    weekdays = np.asarray([draw.draw_date.weekday() for draw in ordered], dtype=int)
    chances = np.asarray([draw.chance - 1 for draw in ordered], dtype=int)
    prediction_cache = {
        tuple(parameters.values()): _prediction_series(chances, weekdays, parameters)
        for parameters in CHANCE_PARAMETER_GRID
    }
    eligible_dates = np.unique(dates[min_train:])
    folds = [fold for fold in np.array_split(eligible_dates, outer_folds) if len(fold)]
    predicted_blocks: list[np.ndarray] = []
    actual_blocks: list[np.ndarray] = []
    selected_parameters = []
    for test_dates in folds:
        train = np.flatnonzero(dates < test_dates[0])
        test = np.flatnonzero(np.isin(dates, test_dates))
        parameters = _select_parameters(dates, chances, prediction_cache, train)
        selected_parameters.append(parameters)
        predicted_blocks.append(prediction_cache[tuple(parameters.values())][test])
        actual_blocks.append(chances[test])
    predicted = np.concatenate(predicted_blocks)
    actual = np.concatenate(actual_blocks)

    brier_values = _brier_by_draw(predicted, actual)
    log_loss_values = _log_loss_by_draw(predicted, actual)
    brier_deltas = brier_values - uniform_brier(pool)
    log_loss_deltas = log_loss_values - uniform_log_loss(pool)
    top1_hits, tie_sizes = _expected_top1_hits(predicted, actual)
    top1_deltas = top1_hits - uniform_probability(pool)

    rng = np.random.default_rng(seed)
    brier_bootstrap = _moving_block_means(brier_deltas, simulations, block_size, rng)
    brier_p_value = _block_sign_p_value(brier_deltas, simulations, block_size, rng)
    log_loss_bootstrap = _moving_block_means(
        log_loss_deltas, simulations, block_size, rng
    )
    log_loss_p_value = _block_sign_p_value(
        log_loss_deltas, simulations, block_size, rng
    )
    top1_bootstrap = _moving_block_means(top1_deltas, simulations, block_size, rng)
    top1_p_value, top1_method = _top1_p_value(
        top1_hits, tie_sizes, pool, simulations, rng
    )

    brier_low, brier_high = np.quantile(brier_bootstrap, (0.025, 0.975))
    log_loss_low, log_loss_high = np.quantile(log_loss_bootstrap, (0.025, 0.975))
    top1_low, top1_high = np.quantile(top1_bootstrap, (0.025, 0.975))
    adjusted_brier = min(1.0, CHANCE_HYPOTHESES * brier_p_value)
    adjusted_log_loss = min(1.0, CHANCE_HYPOTHESES * log_loss_p_value)
    adjusted_top1 = min(1.0, CHANCE_HYPOTHESES * top1_p_value)

    brier_qualified = bool(brier_high < 0 and adjusted_brier < 0.05)
    log_loss_qualified = bool(log_loss_high < 0 and adjusted_log_loss < 0.05)
    top1_qualified = bool(top1_low > 0 and adjusted_top1 < 0.05)
    basis = "+".join(
        name
        for name, flag in (
            ("brier", brier_qualified),
            ("log_loss", log_loss_qualified),
            ("top1", top1_qualified),
        )
        if flag
    )
    calibration_error, calibration_bins = _calibration(predicted, actual)
    uniform_calibration_error, _ = _calibration(
        np.full_like(predicted, uniform_probability(pool)), actual
    )
    final_parameters = _select_parameters(
        dates, chances, prediction_cache, np.arange(len(ordered))
    )
    return ChanceMLBacktestResult(
        game=game,
        observations=len(ordered),
        test_observations=len(actual),
        outer_folds=len(folds),
        selected_parameters=tuple(selected_parameters),
        final_parameters=final_parameters,
        mean_brier=float(brier_values.mean()),
        uniform_brier=uniform_brier(pool),
        mean_brier_delta=float(brier_deltas.mean()),
        delta_ci_low=float(brier_low),
        delta_ci_high=float(brier_high),
        permutation_p_value=brier_p_value,
        adjusted_brier_p_value=adjusted_brier,
        mean_log_loss=float(log_loss_values.mean()),
        uniform_log_loss=uniform_log_loss(pool),
        mean_log_loss_delta=float(log_loss_deltas.mean()),
        log_loss_ci_low=float(log_loss_low),
        log_loss_ci_high=float(log_loss_high),
        log_loss_p_value=log_loss_p_value,
        adjusted_log_loss_p_value=adjusted_log_loss,
        calibration_error=calibration_error,
        uniform_calibration_error=uniform_calibration_error,
        calibration_bins=calibration_bins,
        top1_accuracy=float(top1_hits.mean()),
        uniform_top1_accuracy=uniform_probability(pool),
        top1_uplift=float(top1_deltas.mean()),
        top1_ci_low=float(top1_low),
        top1_ci_high=float(top1_high),
        top1_p_value=top1_p_value,
        adjusted_top1_p_value=adjusted_top1,
        top1_test_method=top1_method,
        top1_tied_draws=int(np.count_nonzero(tie_sizes > 1)),
        hypotheses=CHANCE_HYPOTHESES,
        qualification_basis=basis or "none",
        qualified=bool(brier_qualified or log_loss_qualified or top1_qualified),
    )


def _probability_map(probabilities: np.ndarray) -> dict[int, float]:
    return {
        number: float(probabilities[number - 1])
        for number in range(1, DEFAULT_RULES.chance_pool + 1)
    }


def predict_chance(
    draws: list[Draw],
    target_date: date,
    game: str = "loto",
    validation: ChanceMLBacktestResult | None = None,
) -> dict[str, object]:
    """Distribution predictive du prochain numero Chance.

    Tant que la validation hors-echantillon ne bat pas l'uniforme, la sortie
    publiee est une abstention explicite: dix probabilites a 10 % et aucun numero.
    La distribution du modele reste consultable dans le bloc `experimental`, qui
    n'est jamais une prediction qualifiee.
    """
    pool = DEFAULT_RULES.chance_pool
    ordered = _ordered_game_draws(draws, game)
    if target_date <= ordered[-1].draw_date:
        raise ValueError("La prediction Chance exige une date future")
    parameters = (
        validation.final_parameters
        if validation is not None
        else DEFAULT_CHANCE_PARAMETERS
    )
    chances = np.asarray([draw.chance - 1 for draw in ordered], dtype=int)
    weekdays = np.asarray([draw.draw_date.weekday() for draw in ordered], dtype=int)
    probabilities = _predict_after_history(
        chances, weekdays, target_date.weekday(), parameters
    )
    selected, tied = argmax_with_ties(probabilities)
    experimental = {
        "number": selected,
        "tied_candidates": [int(value) for value in tied],
        "probabilities": _probability_map(probabilities),
        "probability_spread": float(np.ptp(probabilities)),
        "parameters": dict(parameters),
        "selection_method": "temporal_dirichlet_argmax",
        "disclaimer": (
            "Sortie experimentale issue d'un modele non qualifie: ce n'est pas une "
            "prediction et elle ne revendique aucun avantage sur l'uniforme."
        ),
    }
    qualified = validation is not None and validation.qualified
    if qualified:
        published = probabilities
        status = "qualified"
        method = "temporal_dirichlet_argmax"
        number: int | None = selected
        reason = None
    else:
        published = np.full(pool, uniform_probability(pool))
        status = "abstention"
        method = "uniform_abstention"
        number = None
        reason = (
            "Aucune distribution Chance ne bat l'uniforme en Brier, log-loss ou "
            "Top-1 avec un intervalle a 95% et une correction pour tests multiples."
            if validation is not None
            else "Aucune validation hors-echantillon n'a ete fournie."
        )
    return {
        "status": status,
        "number": number,
        "probabilities": _probability_map(published),
        "probability_sum": float(published.sum()),
        "uniform_probability": uniform_probability(pool),
        "selection_method": method,
        "reason": reason,
        "experimental": experimental,
        "validation": validation.to_dict() if validation is not None else None,
    }
