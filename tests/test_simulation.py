import json
import tempfile
import unittest
from pathlib import Path

from loto_lab.simulation import load_payouts, simulate_bankroll


class SimulationTests(unittest.TestCase):
    def test_load_and_simulate(self) -> None:
        payload = {"payouts": {str(rank): float(10 - rank) for rank in range(1, 10)}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payouts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            payouts = load_payouts(path)
        result = simulate_bankroll(payouts, tickets_per_draw=2, draws_per_run=5, runs=20, seed=3)
        self.assertEqual(result.cost_per_run, 22)
        self.assertGreaterEqual(result.positive_run_rate, 0)
        self.assertLessEqual(result.positive_run_rate, 1)


if __name__ == "__main__":
    unittest.main()
