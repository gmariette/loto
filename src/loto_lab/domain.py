from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class LotteryRules:
    main_pool: int = 49
    main_drawn: int = 5
    chance_pool: int = 10
    ticket_price: float = 2.20
    global_return_rate: float = 0.5435
    second_draw_price: float = 0.80
    second_draw_return_rate: float = 0.59

    def validate_main(self, numbers: tuple[int, ...]) -> tuple[int, ...]:
        normalized = tuple(sorted(numbers))
        if len(normalized) != self.main_drawn:
            raise ValueError(f"Il faut exactement {self.main_drawn} numeros")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Les numeros principaux doivent etre distincts")
        if normalized[0] < 1 or normalized[-1] > self.main_pool:
            raise ValueError(f"Les numeros doivent etre compris entre 1 et {self.main_pool}")
        return normalized

    def validate_chance(self, chance: int) -> int:
        if not 1 <= chance <= self.chance_pool:
            raise ValueError(f"Le numero Chance doit etre compris entre 1 et {self.chance_pool}")
        return chance


DEFAULT_RULES = LotteryRules()


@dataclass(frozen=True, slots=True)
class PrizeResult:
    rank: int
    winners: int | None
    payout: float | None

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("Le rang doit etre positif")
        if self.winners is not None and self.winners < 0:
            raise ValueError("Le nombre de gagnants ne peut pas etre negatif")
        if self.payout is not None and self.payout < 0:
            raise ValueError("Le rapport ne peut pas etre negatif")


@dataclass(frozen=True, slots=True)
class Draw:
    main: tuple[int, ...]
    chance: int
    draw_date: date | None = None
    game: str = "loto"
    prizes: tuple[PrizeResult, ...] = ()
    code_winners: int | None = None
    code_payout: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "main", DEFAULT_RULES.validate_main(self.main))
        DEFAULT_RULES.validate_chance(self.chance)
        if not self.game:
            raise ValueError("Le type de jeu ne peut pas etre vide")
        if self.code_winners is not None and self.code_winners < 0:
            raise ValueError("Le nombre de codes gagnants ne peut pas etre negatif")
        if self.code_payout is not None and self.code_payout < 0:
            raise ValueError("Le rapport des codes ne peut pas etre negatif")


@dataclass(frozen=True, slots=True)
class Ticket:
    main: tuple[int, ...]
    chance: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "main", DEFAULT_RULES.validate_main(self.main))
        DEFAULT_RULES.validate_chance(self.chance)


def winning_rank(ticket: Ticket, draw: Draw) -> int | None:
    matches = len(set(ticket.main).intersection(draw.main))
    chance_match = ticket.chance == draw.chance
    ranks = {
        (5, True): 1,
        (5, False): 2,
        (4, True): 3,
        (4, False): 4,
        (3, True): 5,
        (3, False): 6,
        (2, True): 7,
        (2, False): 8,
        (1, True): 9,
        (0, True): 9,
    }
    return ranks.get((matches, chance_match))
