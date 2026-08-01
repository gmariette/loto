import io
import ssl
import subprocess
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

from loto_lab.data import (
    download_latest_archive,
    load_draws,
    load_draws_many,
    load_legacy_draws,
)

CSV = """date_de_tirage;boule_1;boule_2;boule_3;boule_4;boule_5;numero_chance
03/01/2024;1;2;3;4;5;6
01/01/2024;7;8;9;10;11;2
"""

FDJ_CSV_WITH_DECIMALS = (
    "date_de_tirage;boule_1;boule_2;boule_3;boule_4;boule_5;"
    "numero_chance;rapport_du_rang1\n"
    "29/07/2026;41;34;15;27;40;9;4000000,00\n"
)

FDJ_CSV_WITH_CODES = (
    "date_de_tirage;boule_1;boule_2;boule_3;boule_4;boule_5;numero_chance;"
    "nombre_de_codes_gagnants;rapport_codes_gagnants\n"
    "29/07/2026;41;34;15;27;40;9;10;20000,00\n"
)

LEGACY_CSV = (
    "date_de_tirage;boule_1;boule_2;boule_3;boule_4;boule_5;boule_6;"
    "boule_complementaire\n"
    "01/01/1996;1;2;3;4;5;6;7\n"
)


class DataTests(unittest.TestCase):
    def test_loads_and_sorts_fdj_style_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draws.csv"
            path.write_text(CSV, encoding="utf-8")
            draws = load_draws(path)
        self.assertEqual(len(draws), 2)
        self.assertEqual(draws[0].main, (7, 8, 9, 10, 11))
        self.assertEqual(draws[-1].chance, 6)

    def test_loads_csv_inside_zip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draws.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("resultats.csv", CSV)
            draws = load_draws(path)
        self.assertEqual(len(draws), 2)

    def test_fdj_decimal_commas_do_not_confuse_delimiter_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fdj.csv"
            path.write_text(FDJ_CSV_WITH_DECIMALS, encoding="utf-8")
            draws = load_draws(path)
        self.assertEqual(draws[0].main, (15, 27, 34, 40, 41))
        self.assertEqual(draws[0].chance, 9)
        self.assertEqual(draws[0].prizes[0].payout, 4_000_000)

    def test_loads_code_prizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fdj.csv"
            path.write_text(FDJ_CSV_WITH_CODES, encoding="utf-8")
            draw = load_draws(path)[0]
        self.assertEqual(draw.code_winners, 10)
        self.assertEqual(draw.code_payout, 20_000)

    def test_rejects_invalid_draw(self) -> None:
        invalid = CSV.replace("1;2;3;4;5;6", "1;1;3;4;5;6")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text(invalid, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "distincts"):
                load_draws(path)

    def test_loads_many_current_archives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            first.write_text(CSV, encoding="utf-8")
            second.write_text(CSV, encoding="utf-8")
            draws = load_draws_many((first, second))
        self.assertEqual(len(draws), 2)

    def test_loads_legacy_regime_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            path.write_text(LEGACY_CSV, encoding="utf-8")
            draws = load_legacy_draws(path, from_year=1996)
        self.assertEqual(draws[0].main, (1, 2, 3, 4, 5, 6))
        self.assertEqual(draws[0].complementary, 7)

    def test_game_is_inferred_from_archive_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "super-loto-test.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("superloto.csv", CSV)
            draws = load_draws(path)
        self.assertEqual(draws[0].game, "super_loto")

    def test_download_falls_back_to_verified_curl_on_python_tls_error(self) -> None:
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.writestr("draws.csv", CSV)
        tls_error = urllib.error.URLError(ssl.SSLCertVerificationError(1, "bad chain"))
        completed = subprocess.CompletedProcess([], 0, stdout=archive_bytes.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "latest.zip"
            with (
                patch("loto_lab.data.urllib.request.urlopen", side_effect=tls_error),
                patch("loto_lab.data.shutil.which", return_value="/usr/bin/curl"),
                patch("loto_lab.data.subprocess.run", return_value=completed) as run,
            ):
                download_latest_archive(target, "https://example.test/archive")
        command = run.call_args.args[0]
        self.assertNotIn("--insecure", command)
        self.assertIn("--fail", command)


if __name__ == "__main__":
    unittest.main()
