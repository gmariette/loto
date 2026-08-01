from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from math import comb, exp, lgamma, log, log1p, sqrt

from .domain import DEFAULT_RULES, Draw, LotteryRules


@dataclass(frozen=True, slots=True)
class UniformityResult:
    observations: int
    chi_square: float
    monte_carlo_p_value: float
    expected_count: float
    counts: dict[int, int]
    z_scores: dict[int, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _chi_square(counts: list[int], expected: float) -> float:
    return sum((count - expected) ** 2 / expected for count in counts)


def main_uniformity(
    draws: list[Draw], simulations: int = 2_000, seed: int = 0,
    rules: LotteryRules = DEFAULT_RULES,
) -> UniformityResult:
    if not draws:
        raise ValueError("Au moins un tirage est requis")
    if simulations < 1:
        raise ValueError("Le nombre de simulations doit etre positif")
    counts = Counter(number for draw in draws for number in draw.main)
    expected = len(draws) * rules.main_drawn / rules.main_pool
    observed_values = [counts[number] for number in range(1, rules.main_pool + 1)]
    observed_stat = _chi_square(observed_values, expected)

    rng = random.Random(seed)
    extreme = 0
    population = range(1, rules.main_pool + 1)
    for _ in range(simulations):
        simulated = [0] * rules.main_pool
        for _ in draws:
            for number in rng.sample(population, rules.main_drawn):
                simulated[number - 1] += 1
        if _chi_square(simulated, expected) >= observed_stat:
            extreme += 1
    p_value = (extreme + 1) / (simulations + 1)

    variance = len(draws) * (rules.main_drawn / rules.main_pool) * (
        1 - rules.main_drawn / rules.main_pool
    )
    z_scores = {
        number: (counts[number] - expected) / sqrt(variance)
        for number in range(1, rules.main_pool + 1)
    }
    return UniformityResult(
        len(draws), observed_stat, p_value, expected, dict(sorted(counts.items())), z_scores
    )


def chance_uniformity(
    draws: list[Draw], simulations: int = 10_000, seed: int = 0,
    rules: LotteryRules = DEFAULT_RULES,
) -> UniformityResult:
    if not draws:
        raise ValueError("Au moins un tirage est requis")
    counts = Counter(draw.chance for draw in draws)
    expected = len(draws) / rules.chance_pool
    values = [counts[number] for number in range(1, rules.chance_pool + 1)]
    observed_stat = _chi_square(values, expected)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(simulations):
        simulated = [0] * rules.chance_pool
        for _ in draws:
            simulated[rng.randrange(rules.chance_pool)] += 1
        if _chi_square(simulated, expected) >= observed_stat:
            extreme += 1
    variance = len(draws) * (1 / rules.chance_pool) * (1 - 1 / rules.chance_pool)
    return UniformityResult(
        len(draws),
        observed_stat,
        (extreme + 1) / (simulations + 1),
        expected,
        dict(sorted(counts.items())),
        {
            number: (counts[number] - expected) / sqrt(variance)
            for number in range(1, rules.chance_pool + 1)
        },
    )


def lag_overlap(draws: list[Draw], rules: LotteryRules = DEFAULT_RULES) -> dict[str, float]:
    if len(draws) < 2:
        raise ValueError("Au moins deux tirages sont requis")
    overlaps = [
        len(set(previous.main).intersection(current.main))
        for previous, current in zip(draws, draws[1:], strict=False)
    ]
    observed = sum(overlaps) / len(overlaps)
    expected = rules.main_drawn**2 / rules.main_pool
    return {
        "transitions": len(overlaps),
        "observed_mean": observed,
        "expected_mean": expected,
        "difference": observed - expected,
    }


def pair_frequency_outliers(
    draws: list[Draw], limit: int = 10, rules: LotteryRules = DEFAULT_RULES
) -> list[dict[str, float | int | tuple[int, int]]]:
    if not draws:
        raise ValueError("Au moins un tirage est requis")
    counts = Counter(pair for draw in draws for pair in combinations(draw.main, 2))
    probability = comb(rules.main_pool - 2, rules.main_drawn - 2) / comb(
        rules.main_pool, rules.main_drawn
    )
    expected = len(draws) * probability
    standard_deviation = sqrt(len(draws) * probability * (1 - probability))
    results = []
    for pair in combinations(range(1, rules.main_pool + 1), 2):
        count = counts[pair]
        results.append(
            {
                "pair": pair,
                "count": count,
                "expected": expected,
                "z_score": (count - expected) / standard_deviation,
            }
        )
    selected = sorted(results, key=lambda item: abs(float(item["z_score"])), reverse=True)[
        :limit
    ]
    tests = comb(rules.main_pool, 2)
    for item in selected:
        count = int(item["count"])
        lower = _binomial_range_probability(len(draws), probability, 0, count)
        upper = _binomial_range_probability(len(draws), probability, count, len(draws))
        p_value = min(1.0, 2 * min(lower, upper))
        item["p_value_two_sided"] = p_value
        item["bonferroni_p_value"] = min(1.0, p_value * tests)
    return selected


def _binomial_range_probability(n: int, probability: float, start: int, end: int) -> float:
    total = 0.0
    for successes in range(start, end + 1):
        log_probability = (
            lgamma(n + 1)
            - lgamma(successes + 1)
            - lgamma(n - successes + 1)
            + successes * log(probability)
            + (n - successes) * log1p(-probability)
        )
        total += exp(log_probability)
    return min(1.0, total)
