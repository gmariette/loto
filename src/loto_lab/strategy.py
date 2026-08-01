from __future__ import annotations

import random
from math import log

from .domain import DEFAULT_RULES, LotteryRules, Ticket


def anti_crowd_score(numbers: tuple[int, ...]) -> float:
    """Heuristique de partage, sans effet sur la probabilite de sortie."""
    ordered = tuple(sorted(numbers))
    score = 1.2 * sum(number > 31 for number in ordered)
    if all(number <= 31 for number in ordered):
        score -= 5.0
    score -= 1.5 * sum(right - left == 1 for left, right in zip(ordered, ordered[1:], strict=False))
    if len({ordered[index + 1] - ordered[index] for index in range(4)}) == 1:
        score -= 4.0
    repeated_endings = 5 - len({number % 10 for number in ordered})
    score -= 0.5 * repeated_endings
    score -= 0.4 * sum(number in {7, 13} for number in ordered)
    return score


def _weighted_sample_without_replacement(
    rng: random.Random, weights: list[float], count: int
) -> tuple[int, ...]:
    if len(weights) < count or any(weight <= 0 for weight in weights):
        raise ValueError("Les poids doivent etre strictement positifs et assez nombreux")
    keys = [(log(rng.random()) / weight, index + 1) for index, weight in enumerate(weights)]
    return tuple(sorted(number for _, number in sorted(keys, reverse=True)[:count]))


def _diversity_score(candidate: tuple[int, ...], existing: list[Ticket]) -> float:
    if not existing:
        return 1.0
    candidate_set = set(candidate)
    max_overlap = max(len(candidate_set.intersection(ticket.main)) for ticket in existing)
    return 1 - max_overlap / len(candidate)


def generate_tickets(
    count: int,
    seed: int | None = None,
    weights: list[float] | None = None,
    anti_crowd: bool = True,
    candidates_per_ticket: int = 300,
    rules: LotteryRules = DEFAULT_RULES,
) -> list[Ticket]:
    if count < 1:
        raise ValueError("Le nombre de grilles doit etre positif")
    if candidates_per_ticket < 1:
        raise ValueError("candidates_per_ticket doit etre positif")
    weights = weights or [1.0] * rules.main_pool
    if len(weights) != rules.main_pool:
        raise ValueError(f"Il faut {rules.main_pool} poids")
    rng = random.Random(seed)
    tickets: list[Ticket] = []
    chances = list(range(1, rules.chance_pool + 1))
    rng.shuffle(chances)

    for index in range(count):
        best: tuple[float, tuple[int, ...]] | None = None
        seen = {ticket.main for ticket in tickets}
        for _ in range(candidates_per_ticket):
            candidate = _weighted_sample_without_replacement(rng, weights, rules.main_drawn)
            if candidate in seen:
                continue
            crowd_component = anti_crowd_score(candidate) if anti_crowd else 0.0
            score = crowd_component + 3.0 * _diversity_score(candidate, tickets)
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is None:
            raise RuntimeError("Impossible de produire assez de grilles distinctes")
        chance = chances[index % rules.chance_pool]
        tickets.append(Ticket(best[1], chance))
    return tickets
