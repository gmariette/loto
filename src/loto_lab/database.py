from __future__ import annotations

import hashlib
import sqlite3
from datetime import date
from pathlib import Path

from .data import OFFICIAL_ARCHIVES, LegacyDraw, load_draws, load_legacy_draws
from .domain import Draw, PrizeResult

LEGACY_ARCHIVES = {"loto-1976-2008.zip", "super-loto-1996-2008.zip"}

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    filename TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    draw_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS draws (
    id INTEGER PRIMARY KEY,
    game TEXT NOT NULL,
    regime TEXT NOT NULL CHECK (regime IN ('5/49+chance', '6/49+complementary')),
    draw_date TEXT NOT NULL,
    extra_number INTEGER NOT NULL,
    main_key TEXT NOT NULL,
    source_file TEXT NOT NULL REFERENCES sources(filename),
    UNIQUE (game, regime, draw_date, main_key, extra_number)
);

CREATE TABLE IF NOT EXISTS draw_numbers (
    draw_id INTEGER NOT NULL REFERENCES draws(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    number INTEGER NOT NULL CHECK (number BETWEEN 1 AND 49),
    PRIMARY KEY (draw_id, position),
    UNIQUE (draw_id, number)
);

CREATE TABLE IF NOT EXISTS prizes (
    draw_id INTEGER NOT NULL REFERENCES draws(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    winners INTEGER,
    payout REAL,
    PRIMARY KEY (draw_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_draws_date ON draws(draw_date);
CREATE INDEX IF NOT EXISTS idx_draws_game_regime ON draws(game, regime);
CREATE INDEX IF NOT EXISTS idx_draw_numbers_number ON draw_numbers(number);
"""


def connect_database(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database(path: str | Path) -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_database(target)
    connection.executescript(SCHEMA)
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', '1')"
    )
    connection.commit()
    return connection


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upsert_draw(
    connection: sqlite3.Connection,
    *,
    game: str,
    regime: str,
    draw_date: date,
    main: tuple[int, ...],
    extra_number: int,
    source_file: str,
    prizes: tuple[PrizeResult, ...],
) -> int:
    main_key = "-".join(str(number) for number in sorted(main))
    connection.execute(
        """
        INSERT INTO draws(game, regime, draw_date, extra_number, main_key, source_file)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(game, regime, draw_date, main_key, extra_number)
        DO UPDATE SET source_file = excluded.source_file
        """,
        (game, regime, draw_date.isoformat(), extra_number, main_key, source_file),
    )
    row = connection.execute(
        """
        SELECT id FROM draws
        WHERE game = ? AND regime = ? AND draw_date = ? AND main_key = ? AND extra_number = ?
        """,
        (game, regime, draw_date.isoformat(), main_key, extra_number),
    ).fetchone()
    draw_id = int(row["id"])
    connection.execute("DELETE FROM draw_numbers WHERE draw_id = ?", (draw_id,))
    connection.executemany(
        "INSERT INTO draw_numbers(draw_id, position, number) VALUES (?, ?, ?)",
        ((draw_id, position, number) for position, number in enumerate(main, start=1)),
    )
    connection.execute("DELETE FROM prizes WHERE draw_id = ?", (draw_id,))
    connection.executemany(
        "INSERT INTO prizes(draw_id, rank, winners, payout) VALUES (?, ?, ?, ?)",
        ((draw_id, prize.rank, prize.winners, prize.payout) for prize in prizes),
    )
    return draw_id


def import_current_archive(connection: sqlite3.Connection, path: str | Path) -> int:
    source = Path(path)
    draws = load_draws(source)
    connection.execute(
        """
        INSERT INTO sources(filename, sha256, imported_at, draw_count)
        VALUES (?, ?, CURRENT_TIMESTAMP, ?)
        ON CONFLICT(filename) DO UPDATE SET
            sha256 = excluded.sha256,
            imported_at = CURRENT_TIMESTAMP,
            draw_count = excluded.draw_count
        """,
        (source.name, _checksum(source), len(draws)),
    )
    for draw in draws:
        if draw.draw_date is None:
            continue
        _upsert_draw(
            connection,
            game=draw.game,
            regime="5/49+chance",
            draw_date=draw.draw_date,
            main=draw.main,
            extra_number=draw.chance,
            source_file=source.name,
            prizes=draw.prizes,
        )
    connection.commit()
    return len(draws)


def import_legacy_archive(
    connection: sqlite3.Connection, path: str | Path, from_year: int = 1996
) -> int:
    source = Path(path)
    draws = load_legacy_draws(source, from_year)
    connection.execute(
        """
        INSERT INTO sources(filename, sha256, imported_at, draw_count)
        VALUES (?, ?, CURRENT_TIMESTAMP, ?)
        ON CONFLICT(filename) DO UPDATE SET
            sha256 = excluded.sha256,
            imported_at = CURRENT_TIMESTAMP,
            draw_count = excluded.draw_count
        """,
        (source.name, _checksum(source), len(draws)),
    )
    for draw in draws:
        if draw.draw_date is None:
            continue
        _upsert_draw(
            connection,
            game=draw.game,
            regime="6/49+complementary",
            draw_date=draw.draw_date,
            main=draw.main,
            extra_number=draw.complementary,
            source_file=source.name,
            prizes=draw.prizes,
        )
    connection.commit()
    return len(draws)


def build_database(
    database: str | Path, archives_dir: str | Path, from_year: int = 1996
) -> dict[str, object]:
    directory = Path(archives_dir)
    missing = [name for name in OFFICIAL_ARCHIVES if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError("Archives absentes: " + ", ".join(missing))
    connection = initialize_database(database)
    imported: dict[str, int] = {}
    try:
        for filename in OFFICIAL_ARCHIVES:
            path = directory / filename
            if filename in LEGACY_ARCHIVES:
                imported[filename] = import_legacy_archive(connection, path, from_year)
            else:
                imported[filename] = import_current_archive(connection, path)
        return database_info(connection, imported)
    finally:
        connection.close()


def _load_numbers(connection: sqlite3.Connection, draw_id: int) -> tuple[int, ...]:
    rows = connection.execute(
        "SELECT number FROM draw_numbers WHERE draw_id = ? ORDER BY position", (draw_id,)
    )
    return tuple(int(row["number"]) for row in rows)


def _load_prizes(connection: sqlite3.Connection, draw_id: int) -> tuple[PrizeResult, ...]:
    rows = connection.execute(
        "SELECT rank, winners, payout FROM prizes WHERE draw_id = ? ORDER BY rank", (draw_id,)
    )
    return tuple(
        PrizeResult(
            int(row["rank"]),
            int(row["winners"]) if row["winners"] is not None else None,
            float(row["payout"]) if row["payout"] is not None else None,
        )
        for row in rows
    )


def load_current_database(path: str | Path) -> list[Draw]:
    connection = connect_database(path)
    try:
        rows = connection.execute(
            """
            SELECT id, game, draw_date, extra_number
            FROM draws WHERE regime = '5/49+chance'
            ORDER BY draw_date, game, id
            """
        ).fetchall()
        return [
            Draw(
                _load_numbers(connection, int(row["id"])),
                int(row["extra_number"]),
                date.fromisoformat(str(row["draw_date"])),
                str(row["game"]),
                _load_prizes(connection, int(row["id"])),
            )
            for row in rows
        ]
    finally:
        connection.close()


def load_legacy_database(path: str | Path, from_year: int | None = None) -> list[LegacyDraw]:
    connection = connect_database(path)
    try:
        parameters: tuple[object, ...] = ()
        where = "regime = '6/49+complementary'"
        if from_year is not None:
            where += " AND draw_date >= ?"
            parameters = (f"{from_year:04d}-01-01",)
        rows = connection.execute(
            f"""
            SELECT id, game, draw_date, extra_number
            FROM draws WHERE {where}
            ORDER BY draw_date, game, id
            """,  # noqa: S608
            parameters,
        ).fetchall()
        return [
            LegacyDraw(
                _load_numbers(connection, int(row["id"])),
                int(row["extra_number"]),
                date.fromisoformat(str(row["draw_date"])),
                str(row["game"]),
                _load_prizes(connection, int(row["id"])),
            )
            for row in rows
        ]
    finally:
        connection.close()


def database_info(
    connection_or_path: sqlite3.Connection | str | Path,
    imported: dict[str, int] | None = None,
) -> dict[str, object]:
    owns_connection = not isinstance(connection_or_path, sqlite3.Connection)
    connection = (
        connect_database(connection_or_path) if owns_connection else connection_or_path
    )
    try:
        rows = connection.execute("SELECT game, regime, COUNT(*) AS count FROM draws GROUP BY 1, 2")
        counts = {f"{row['game']}:{row['regime']}": int(row["count"]) for row in rows}
        bounds = connection.execute(
            """
            SELECT MIN(draw_date) AS first_date,
                   MAX(draw_date) AS last_date,
                   COUNT(*) AS count
            FROM draws
            """
        ).fetchone()
        return {
            "draws": int(bounds["count"]),
            "first_date": bounds["first_date"],
            "last_date": bounds["last_date"],
            "by_game_and_regime": counts,
            "sources": connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
            "prize_rows": connection.execute("SELECT COUNT(*) FROM prizes").fetchone()[0],
            "imported": imported,
        }
    finally:
        if owns_connection:
            connection.close()
