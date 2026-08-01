from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path

from .data import (
    download_all_archives,
    download_latest_archive,
    load_draws,
    load_draws_many,
    load_legacy_draws_many,
)
from .database import build_database, database_info
from .domain import DEFAULT_RULES, LotteryRules
from .ml import nested_ml_backtest, predict_next_draw
from .models import SmoothedFrequencyPredictor, standard_backtests
from .probability import (
    expected_budget,
    probability_of_any_prize,
    rank_probabilities,
    total_outcomes,
)
from .simulation import load_payouts, simulate_bankroll
from .stats import chance_uniformity, lag_overlap, main_uniformity, pair_frequency_outliers
from .strategy import anti_crowd_score, generate_tickets
from .value import estimate_value


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def command_odds(args: argparse.Namespace) -> None:
    stakes = args.grids * DEFAULT_RULES.ticket_price
    result = {
        "rules": asdict(DEFAULT_RULES),
        "total_outcomes": total_outcomes(),
        "any_prize_probability": probability_of_any_prize(),
        "one_prize_in": 1 / probability_of_any_prize(),
        "ranks": [rank.to_dict() for rank in rank_probabilities()],
        "budget": expected_budget(stakes),
    }
    _print_json(result)


def command_download(args: argparse.Namespace) -> None:
    target = download_latest_archive(args.output, args.url)
    draws = load_draws(target)
    print(f"Archive enregistree: {target} ({len(draws)} tirages valides)")


def command_download_all(args: argparse.Namespace) -> None:
    targets = download_all_archives(args.output_dir)
    print(f"{len(targets)} archives enregistrees dans {args.output_dir}")


def command_build_db(args: argparse.Namespace) -> None:
    _print_json(build_database(args.db, args.archives_dir, args.from_year))


def command_db_info(args: argparse.Namespace) -> None:
    _print_json(database_info(args.db))


def command_analyze(args: argparse.Namespace) -> None:
    draws = load_draws_many(args.data)
    result = {
        "draws": len(draws),
        "first_date": draws[0].draw_date if draws else None,
        "last_date": draws[-1].draw_date if draws else None,
        "games": dict(Counter(draw.game for draw in draws)),
        "main_uniformity": main_uniformity(draws, args.simulations, args.seed).to_dict(),
        "chance_uniformity": chance_uniformity(draws, args.simulations, args.seed).to_dict(),
        "lag_overlap": lag_overlap(draws),
        "largest_pair_deviations": pair_frequency_outliers(draws),
        "interpretation": (
            "Une petite p-value signale un ecart a investiguer; elle ne demontre ni causalite "
            "ni capacite predictive. Confirmer sur une periode future separee."
        ),
    }
    _print_json(result)


def command_analyze_legacy(args: argparse.Namespace) -> None:
    draws = load_legacy_draws_many(args.data, args.from_year)
    legacy_rules = LotteryRules(main_drawn=6)
    result = {
        "regime": "6 numeros sur 49 + complementaire, non comparable au Loto actuel",
        "draws": len(draws),
        "first_date": draws[0].draw_date if draws else None,
        "last_date": draws[-1].draw_date if draws else None,
        "games": dict(Counter(draw.game for draw in draws)),
        "main_uniformity": main_uniformity(
            draws, args.simulations, args.seed, legacy_rules
        ).to_dict(),
        "lag_overlap": lag_overlap(draws, legacy_rules),
        "largest_pair_deviations": pair_frequency_outliers(draws, rules=legacy_rules),
    }
    _print_json(result)


def command_backtest(args: argparse.Namespace) -> None:
    draws = load_draws_many(args.data)
    _print_json([result.to_dict() for result in standard_backtests(draws, args.min_train)])


def command_ml_backtest(args: argparse.Namespace) -> None:
    draws = load_draws_many(args.data)
    results = nested_ml_backtest(
        draws,
        min_history=args.min_history,
        min_train=args.min_train,
        outer_folds=args.folds,
        simulations=args.simulations,
        seed=args.seed,
    )
    payload = {
        "results": [result.to_dict() for result in results],
        "decision": (
            "model_available" if any(result.qualified for result in results) else "abstention"
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    _print_json(payload)


def command_ml_predict(args: argparse.Namespace) -> None:
    draws = load_draws_many(args.data)
    results = nested_ml_backtest(
        draws,
        min_history=args.min_history,
        min_train=args.min_train,
        outer_folds=args.folds,
        simulations=args.simulations,
        seed=args.seed,
    )
    _print_json(
        predict_next_draw(
            draws,
            results,
            date.fromisoformat(args.date),
            game=args.game,
            force=args.force,
            seed=args.seed,
        )
    )


def command_value(args: argparse.Namespace) -> None:
    draws = load_draws_many(args.data)
    _print_json(
        estimate_value(
            draws,
            jackpot=args.jackpot,
            game=args.game,
            expected_co_winners=args.co_winners,
            bootstrap_simulations=args.simulations,
            seed=args.seed,
        ).to_dict()
    )


def command_generate(args: argparse.Namespace) -> None:
    weights = None
    if args.mode == "frequency":
        if not args.data:
            raise SystemExit("--data est obligatoire pour le mode frequency")
        draws = load_draws_many(args.data)
        weights = SmoothedFrequencyPredictor(alpha=20).predict(draws, DEFAULT_RULES)
    tickets = generate_tickets(
        args.count,
        seed=args.seed,
        weights=weights,
        anti_crowd=args.mode == "anti-crowd",
    )
    _print_json(
        [
            {
                "main": ticket.main,
                "chance": ticket.chance,
                "anti_crowd_score": anti_crowd_score(ticket.main),
            }
            for ticket in tickets
        ]
    )


def command_simulate(args: argparse.Namespace) -> None:
    payouts = load_payouts(args.payouts)
    result = simulate_bankroll(
        payouts,
        tickets_per_draw=args.tickets,
        draws_per_run=args.draws,
        runs=args.runs,
        seed=args.seed,
    )
    _print_json(result.to_dict())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loto-lab",
        description="Analyse reproductible du Loto FDJ, sans promesse de prediction.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    odds = subparsers.add_parser("odds", help="Calculer les probabilites exactes")
    odds.add_argument("--grids", type=int, default=1)
    odds.set_defaults(handler=command_odds)

    download = subparsers.add_parser("download", help="Telecharger l'archive FDJ recente")
    download.add_argument("--output", type=Path, default=Path("data/loto-latest.zip"))
    download.add_argument("--url", default=None)
    download.set_defaults(handler=command_download)

    download_all = subparsers.add_parser(
        "download-all", help="Telecharger toutes les archives officielles depuis 1976"
    )
    download_all.add_argument("--output-dir", type=Path, default=Path("data"))
    download_all.set_defaults(handler=command_download_all)

    build_db = subparsers.add_parser(
        "build-db", help="Construire ou actualiser la base SQLite normalisee"
    )
    build_db.add_argument("--db", type=Path, default=Path("data/loto.sqlite"))
    build_db.add_argument("--archives-dir", type=Path, default=Path("data"))
    build_db.add_argument("--from-year", type=int, default=1996)
    build_db.set_defaults(handler=command_build_db)

    db_info = subparsers.add_parser("db-info", help="Afficher le contenu de la base SQLite")
    db_info.add_argument("--db", type=Path, default=Path("data/loto.sqlite"))
    db_info.set_defaults(handler=command_db_info)

    analyze = subparsers.add_parser("analyze", help="Tester l'uniformite et les dependances")
    analyze.add_argument("data", type=Path, nargs="+")
    analyze.add_argument("--simulations", type=int, default=2_000)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.set_defaults(handler=command_analyze)

    legacy = subparsers.add_parser(
        "analyze-legacy", help="Analyser separement l'ancien regime 6/49"
    )
    legacy.add_argument("data", type=Path, nargs="+")
    legacy.add_argument("--from-year", type=int, default=1996)
    legacy.add_argument("--simulations", type=int, default=2_000)
    legacy.add_argument("--seed", type=int, default=0)
    legacy.set_defaults(handler=command_analyze_legacy)

    backtest = subparsers.add_parser("backtest", help="Backtest chronologique contre l'uniforme")
    backtest.add_argument("data", type=Path, nargs="+")
    backtest.add_argument("--min-train", type=int, default=200)
    backtest.set_defaults(handler=command_backtest)

    ml_backtest = subparsers.add_parser(
        "ml-backtest", help="Validation temporelle imbriquee des modeles ML"
    )
    ml_backtest.add_argument("data", type=Path, nargs="+")
    ml_backtest.add_argument("--min-history", type=int, default=50)
    ml_backtest.add_argument("--min-train", type=int, default=500)
    ml_backtest.add_argument("--folds", type=int, default=3)
    ml_backtest.add_argument("--simulations", type=int, default=2_000)
    ml_backtest.add_argument("--seed", type=int, default=0)
    ml_backtest.add_argument("--output", type=Path)
    ml_backtest.set_defaults(handler=command_ml_backtest)

    ml_predict = subparsers.add_parser(
        "ml-predict", help="Predire uniquement si un modele est qualifie"
    )
    ml_predict.add_argument("data", type=Path, nargs="+")
    ml_predict.add_argument("--date", required=True)
    ml_predict.add_argument(
        "--game", choices=("loto", "super_loto", "grand_loto"), default="loto"
    )
    ml_predict.add_argument("--min-history", type=int, default=50)
    ml_predict.add_argument("--min-train", type=int, default=500)
    ml_predict.add_argument("--folds", type=int, default=3)
    ml_predict.add_argument("--simulations", type=int, default=2_000)
    ml_predict.add_argument("--seed", type=int, default=0)
    ml_predict.add_argument("--force", action="store_true")
    ml_predict.set_defaults(handler=command_ml_predict)

    value = subparsers.add_parser(
        "value", help="Estimer l'esperance monetaire d'un jackpot annonce"
    )
    value.add_argument("data", type=Path, nargs="+")
    value.add_argument("--jackpot", type=float, required=True)
    value.add_argument(
        "--game", choices=("loto", "super_loto", "grand_loto"), default="loto"
    )
    value.add_argument("--co-winners", type=float, default=0.0)
    value.add_argument("--simulations", type=int, default=2_000)
    value.add_argument("--seed", type=int, default=0)
    value.set_defaults(handler=command_value)

    generate = subparsers.add_parser("generate", help="Generer des grilles distinctes")
    generate.add_argument("--count", type=int, default=5)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument(
        "--mode",
        choices=("uniform", "anti-crowd", "frequency"),
        default="anti-crowd",
    )
    generate.add_argument("--data", type=Path, nargs="+")
    generate.set_defaults(handler=command_generate)

    simulate = subparsers.add_parser("simulate", help="Simuler une bankroll avec gains explicites")
    simulate.add_argument("--payouts", type=Path, required=True)
    simulate.add_argument("--tickets", type=int, default=1)
    simulate.add_argument("--draws", type=int, default=156)
    simulate.add_argument("--runs", type=int, default=10_000)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.set_defaults(handler=command_simulate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "download" and args.url is None:
        from .data import LATEST_ARCHIVE_URL

        args.url = LATEST_ARCHIVE_URL
    args.handler(args)
