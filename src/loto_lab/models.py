from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import erfc, exp, log, sqrt

from .domain import DEFAULT_RULES, Draw, LotteryRules


class Predictor:
    name = "predictor"

    def predict(self, history: list[Draw], rules: LotteryRules) -> list[float]:
        raise NotImplementedError


class UniformPredictor(Predictor):
    name = "uniforme"

    def predict(self, history: list[Draw], rules: LotteryRules) -> list[float]:
        return [rules.main_drawn / rules.main_pool] * rules.main_pool


class SmoothedFrequencyPredictor(Predictor):
    def __init__(self, alpha: float = 20.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha doit etre positif")
        self.alpha = alpha
        self.name = f"frequence_lissee_alpha_{alpha:g}"

    def predict(self, history: list[Draw], rules: LotteryRules) -> list[float]:
        counts = Counter(number for draw in history for number in draw.main)
        denominator = len(history) * rules.main_drawn + self.alpha * rules.main_pool
        return [
            rules.main_drawn * (counts[number] + self.alpha) / denominator
            for number in range(1, rules.main_pool + 1)
        ]


class DecayedFrequencyPredictor(Predictor):
    def __init__(self, half_life: float = 100.0, alpha: float = 5.0) -> None:
        if half_life <= 0 or alpha <= 0:
            raise ValueError("half_life et alpha doivent etre positifs")
        self.half_life = half_life
        self.alpha = alpha
        self.name = f"frequence_decroissante_hl_{half_life:g}"

    def predict(self, history: list[Draw], rules: LotteryRules) -> list[float]:
        weights = [self.alpha] * rules.main_pool
        decay = log(2) / self.half_life
        for age, draw in enumerate(reversed(history)):
            weight = exp(-decay * age)
            for number in draw.main:
                weights[number - 1] += weight
        scale = rules.main_drawn / sum(weights)
        return [weight * scale for weight in weights]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    model: str
    test_draws: int
    mean_brier: float
    uniform_brier: float
    mean_delta: float
    standard_error: float
    z_score: float
    p_value_two_sided: float
    mean_top5_hits: float
    expected_uniform_top5_hits: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _brier(probabilities: list[float], actual: tuple[int, ...]) -> float:
    actual_set = set(actual)
    return sum(
        (probability - (1.0 if number in actual_set else 0.0)) ** 2
        for number, probability in enumerate(probabilities, start=1)
    ) / len(probabilities)


def walk_forward_backtest(
    draws: list[Draw],
    model_factory: Callable[[], Predictor],
    min_train: int = 200,
    rules: LotteryRules = DEFAULT_RULES,
) -> BacktestResult:
    if min_train < 1 or len(draws) <= min_train:
        raise ValueError("Il faut davantage de tirages que min_train")
    uniform = UniformPredictor()
    model_scores: list[float] = []
    uniform_scores: list[float] = []
    top_hits: list[int] = []
    model_name = model_factory().name

    for index in range(min_train, len(draws)):
        actual = draws[index]
        history = [
            draw
            for draw in draws[:index]
            if actual.draw_date is None
            or draw.draw_date is None
            or draw.draw_date < actual.draw_date
        ]
        if len(history) < min_train:
            continue
        probabilities = model_factory().predict(history, rules)
        baseline = uniform.predict(history, rules)
        invalid_size = len(probabilities) != rules.main_pool
        invalid_sum = abs(sum(probabilities) - rules.main_drawn) > 1e-9
        if invalid_size or invalid_sum:
            raise ValueError(
                "Le modele doit produire 49 probabilites marginales dont la somme vaut 5"
            )
        model_scores.append(_brier(probabilities, actual.main))
        uniform_scores.append(_brier(baseline, actual.main))
        top_numbers = sorted(
            range(1, rules.main_pool + 1),
            key=lambda number: probabilities[number - 1],
            reverse=True,
        )[: rules.main_drawn]
        top_hits.append(len(set(top_numbers).intersection(actual.main)))

    deltas = [
        model - baseline
        for model, baseline in zip(model_scores, uniform_scores, strict=True)
    ]
    mean_delta = sum(deltas) / len(deltas)
    if len(deltas) > 1:
        variance = sum((value - mean_delta) ** 2 for value in deltas) / (len(deltas) - 1)
        standard_error = sqrt(variance / len(deltas))
    else:
        standard_error = 0.0
    z_score = mean_delta / standard_error if standard_error else 0.0
    p_value = erfc(abs(z_score) / sqrt(2)) if standard_error else 1.0
    return BacktestResult(
        model=model_name,
        test_draws=len(deltas),
        mean_brier=sum(model_scores) / len(model_scores),
        uniform_brier=sum(uniform_scores) / len(uniform_scores),
        mean_delta=mean_delta,
        standard_error=standard_error,
        z_score=z_score,
        p_value_two_sided=p_value,
        mean_top5_hits=sum(top_hits) / len(top_hits),
        expected_uniform_top5_hits=rules.main_drawn**2 / rules.main_pool,
    )


def standard_backtests(
    draws: list[Draw], min_train: int = 200, rules: LotteryRules = DEFAULT_RULES
) -> list[BacktestResult]:
    factories: list[Callable[[], Predictor]] = [
        lambda: SmoothedFrequencyPredictor(alpha=20),
        lambda: DecayedFrequencyPredictor(half_life=50, alpha=5),
        lambda: DecayedFrequencyPredictor(half_life=200, alpha=5),
    ]
    return [walk_forward_backtest(draws, factory, min_train, rules) for factory in factories]
