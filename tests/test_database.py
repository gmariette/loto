import tempfile
import unittest
from datetime import date
from pathlib import Path

from loto_lab.data import LegacyDraw
from loto_lab.database import (
    database_info,
    import_current_archive,
    initialize_database,
    load_current_database,
)
from loto_lab.domain import Draw


class DatabaseTests(unittest.TestCase):
    def test_current_database_roundtrip(self) -> None:
        csv_content = (
            "date_de_tirage;boule_1;boule_2;boule_3;boule_4;boule_5;numero_chance;"
            "nombre_de_codes_gagnants;rapport_codes_gagnants\n"
            "01/01/2024;1;2;3;4;5;6;10;20000\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "super-loto.csv"
            database = Path(directory) / "loto.sqlite"
            archive.write_text(csv_content, encoding="utf-8")
            connection = initialize_database(database)
            import_current_archive(connection, archive)
            import_current_archive(connection, archive)
            connection.close()
            draws = load_current_database(database)
            info = database_info(database)
        self.assertEqual(
            draws,
            [
                Draw(
                    (1, 2, 3, 4, 5),
                    6,
                    date(2024, 1, 1),
                    "super_loto",
                    code_winners=10,
                    code_payout=20_000,
                )
            ],
        )
        self.assertEqual(info["draws"], 1)
        self.assertEqual(info["code_prize_rows"], 1)

    def test_legacy_draw_validation(self) -> None:
        draw = LegacyDraw((1, 2, 3, 4, 5, 6), 7, date(1996, 1, 1), "loto")
        self.assertEqual(len(draw.main), 6)


if __name__ == "__main__":
    unittest.main()
