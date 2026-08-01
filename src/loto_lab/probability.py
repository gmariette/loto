from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb

from .domain import DEFAULT_RULES, LotteryRules


@dataclass(frozen=True, slots=True)
class RankProbability:
    rank: int
    main_matches: int
    chance_match: bool
    outcomes: int
    probability: float

    @property
    def one_in(self) -> float:
        return 1.0 / self.probability

    def to_dict(self) -> dict[str, int | float | bool]:
        return {**asdict(self), "one_in": self.one_in}


def total_outcomes(rules: LotteryRules = DEFAULT_RULES) -> int:
    return comb(rules.main_pool, rules.main_drawn) * rules.chance_pool


def rank_probabilities(rules: LotteryRules = DEFAULT_RULES) -> list[RankProbability]:
    ranks: list[RankProbability] = []
    denominator = total_outcomes(rules)
    rank_map = {
        (5, True): 1,
        (5, False): 2,
        (4, True): 3,
        (4, False): 4,
        (3, True): 5,
        (3, False): 6,
        (2, True): 7,
        (2, False): 8,
    }
    for (matches, chance_match), rank in rank_map.items():
        main_ways = comb(rules.main_drawn, matches) * comb(
            rules.main_pool - rules.main_drawn, rules.main_drawn - matches
        )
        chance_ways = 1 if chance_match else rules.chance_pool - 1
        outcomes = main_ways * chance_ways
        ranks.append(
            RankProbability(rank, matches, chance_match, outcomes, outcomes / denominator)
        )

    rank_9_outcomes = 0
    for matches in (0, 1):
        rank_9_outcomes += comb(rules.main_drawn, matches) * comb(
            rules.main_pool - rules.main_drawn, rules.main_drawn - matches
        )
    ranks.append(RankProbability(9, -1, True, rank_9_outcomes, rank_9_outcomes / denominator))
    return ranks


def probability_of_any_prize(rules: LotteryRules = DEFAULT_RULES) -> float:
    return sum(item.probability for item in rank_probabilities(rules))


def expected_budget(stakes: float, rules: LotteryRules = DEFAULT_RULES) -> dict[str, float]:
    if stakes < 0:
        raise ValueError("Le montant mise ne peut pas etre negatif")
    expected_return = stakes * rules.global_return_rate
    return {
        "stakes": stakes,
        "expected_return": expected_return,
        "expected_loss": stakes - expected_return,
        "return_rate": rules.global_return_rate,
    }
