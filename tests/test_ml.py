import random
import unittest
from datetime import date, timedelta
from itertools import combinations

import numpy as np

from loto_lab.domain import DEFAULT_RULES, Draw
from loto_lab.ml import (
    DEFAULT_MODELS,
    MODEL_PARAMETERS,
    RANKING_MODELS,
    _expected_top5_hits,
    _fit_predict,
    _holm_values,
    _top5_indices,
    _top5_inference,
    blend_with_uniform,
    build_feature_dataset,
    nested_ml_backtest,
    next_feature_matrix,
    predict_next_draw,
    project_inclusion_probabilities,
    top5_selection,
)


def dated_random_draws(count: int, seed: int = 4) -> list[Draw]:
    rng = random.Random(seed)
    start = date(2020, 1, 1)
    return [
        Draw(
            tuple(rng.sample(range(1, 50), 5)),
            rng.randint(1, 10),
            start + timedelta(days=index),
        )
        for index in range(count)
    ]


def biased_draws(count: int, seed: int = 11) -> list[Draw]:
    """Sequence synthetique ou cinq numeros sortent bien plus souvent que l'uniforme.

    Sert de controle positif: un backtest correct doit detecter ce signal.
    """
    rng = random.Random(seed)
    hot = [1, 2, 3, 4, 5]
    start = date(2020, 1, 1)
    draws = []
    for index in range(count):
        chosen = set(rng.sample(hot, 4))
        while len(chosen) < 5:
            chosen.add(rng.randint(1, 49))
        draws.append(Draw(tuple(sorted(chosen)), rng.randint(1, 10), start + timedelta(days=index)))
    return draws


class MLTests(unittest.TestCase):
    def test_projection_sums_to_five(self) -> None:
        projected = project_inclusion_probabilities(np.linspace(0.01, 0.8, 49))
        self.assertAlmostEqual(float(projected.sum()), 5)
        self.assertTrue(np.all((projected > 0) & (projected < 1)))

    def test_zero_blend_is_exactly_uniform(self) -> None:
        probabilities = np.linspace(0.01, 0.2, 49)[np.newaxis, :]
        blended = blend_with_uniform(probabilities, 0.0)
        np.testing.assert_allclose(blended, 5 / 49)

    def test_top5_selection_is_deterministic_and_flags_ties(self) -> None:
        probabilities = np.full(49, 5 / 49)
        self.assertEqual(_top5_indices(probabilities).tolist(), [0, 1, 2, 3, 4])
        selection = top5_selection(probabilities)
        self.assertEqual(selection["numbers"], (1, 2, 3, 4, 5))
        self.assertEqual(selection["certain_numbers"], ())
        self.assertEqual(len(selection["tied_candidates"]), 49)
        self.assertTrue(selection["selection_is_tied"])

    def test_top5_selection_reports_an_unambiguous_ranking(self) -> None:
        probabilities = np.linspace(0.01, 0.5, 49)
        selection = top5_selection(probabilities)
        self.assertEqual(selection["numbers"], (45, 46, 47, 48, 49))
        self.assertEqual(selection["certain_numbers"], (46, 47, 48, 49))
        self.assertEqual(selection["tied_candidates"], (45,))
        self.assertFalse(selection["selection_is_tied"])

    def test_expected_hits_resolve_ties_without_a_random_seed(self) -> None:
        # Quatre numeros certains et quatre ex aequo pour la derniere place.
        probabilities = np.zeros((1, 49))
        probabilities[0, :4] = 1.0
        probabilities[0, 4:8] = 0.5
        actual = np.zeros((1, 49))
        actual[0, [0, 1, 2, 3, 4]] = 1.0
        hits = _expected_top5_hits(probabilities, actual)
        # 4 certains + 1 place restante partagee entre 4 ex aequo dont 1 gagnant.
        self.assertAlmostEqual(float(hits[0]), 4 + 1 / 4)

    def test_expected_hits_match_exact_counts_without_ties(self) -> None:
        probabilities = np.linspace(0.01, 0.5, 49)[np.newaxis, :]
        actual = np.zeros((1, 49))
        actual[0, [48, 47, 0, 1, 2]] = 1.0
        self.assertAlmostEqual(float(_expected_top5_hits(probabilities, actual)[0]), 2.0)

    def test_top5_inference_detects_large_hit_uplift(self) -> None:
        hits = np.full(100, 5.0)
        structure = (
            np.full(100, DEFAULT_RULES.main_drawn - 1),
            np.ones(100, dtype=int),
            np.ones(100, dtype=int),
        )
        uplift, low, high, p_value = _top5_inference(
            hits, structure, simulations=100, seed=42, block_size=4
        )
        self.assertGreater(uplift, 0)
        self.assertGreater(low, 0)
        self.assertGreater(high, 0)
        self.assertLess(p_value, 0.05)

    def test_top5_null_distribution_is_centred_on_the_uniform_expectation(self) -> None:
        from loto_lab.ml import _top5_null_means

        means = _top5_null_means(
            np.full(200, 4),
            np.ones(200, dtype=int),
            np.ones(200, dtype=int),
            400,
            np.random.default_rng(1),
        )
        self.assertAlmostEqual(float(means.mean()), 25 / 49, places=2)

    def test_no_default_model_is_a_pure_duplicate(self) -> None:
        """Deux modeles ne peuvent coexister que s'ils different vraiment.

        `hierarchical_ridge_ranker` produisait des probabilites bit-a-bit
        identiques a `rolling_ridge_ranker` avec la meme grille et le meme
        objectif de selection: il gonflait la famille de Holm sans rien apporter.
        """
        dataset = build_feature_dataset(dated_random_draws(120), min_history=20)
        predictions: dict[str, bytes] = {}
        for model in DEFAULT_MODELS:
            parameters = {**MODEL_PARAMETERS[model][0], "uniform_blend": 1.0}
            predicted = _fit_predict(
                model, parameters, dataset.x[:60], dataset.y[:60], dataset.x[60:70], 0
            )
            predictions[model] = np.round(predicted, 12).tobytes()
        for left, right in combinations(DEFAULT_MODELS, 2):
            if predictions[left] != predictions[right]:
                continue
            self.assertNotEqual(
                (MODEL_PARAMETERS[left], left in RANKING_MODELS),
                (MODEL_PARAMETERS[right], right in RANKING_MODELS),
                f"{left} et {right} sont indiscernables: memes probabilites, meme "
                "grille et meme objectif de selection",
            )
        self.assertNotIn("hierarchical_ridge_ranker", MODEL_PARAMETERS)

    def test_holm_adjustment_is_monotone(self) -> None:
        adjusted = _holm_values([0.01, 0.02, 0.5])
        np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.5])

    def test_feature_dataset_has_one_row_per_number(self) -> None:
        dataset = build_feature_dataset(dated_random_draws(80), min_history=20)
        self.assertEqual(dataset.x.shape[1], 49)
        self.assertTrue(np.all(dataset.y.sum(axis=1) == 5))

    def test_bayesian_nested_backtest_and_abstention(self) -> None:
        draws = dated_random_draws(180)
        results = nested_ml_backtest(
            draws,
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            models=("bayesian",),
        )
        self.assertEqual(results[0].test_draws, 100)
        self.assertAlmostEqual(results[0].uniform_expected_top5_hits, 25 / 49)
        self.assertGreaterEqual(results[0].top5_p_value, 0.0)
        self.assertLessEqual(results[0].top5_p_value, 1.0)
        self.assertEqual(
            results[0].qualified,
            results[0].probability_qualified or results[0].ranking_qualified,
        )
        prediction = predict_next_draw(draws, results, date(2021, 1, 1))
        if not results[0].qualified:
            self.assertEqual(prediction["status"], "abstention")
            self.assertEqual(prediction["game"], "loto")
            self.assertEqual(prediction["training_last_date"], draws[-1].draw_date)
            forced = predict_next_draw(
                draws, results, date(2021, 1, 1), force=True
            )
            self.assertFalse(forced["validation"]["qualified"])
            self.assertIn("mean_brier_delta", forced["validation"])
            self.assertIn("top5_hit_uplift", forced["validation"])
            self.assertGreaterEqual(forced["probability_spread"], 0.0)

    def test_rolling_bayesian_uses_only_past_window_frequencies(self) -> None:
        draws = dated_random_draws(80)
        dataset = build_feature_dataset(draws, min_history=20)
        parameters = {
            "window": 10,
            "prior_strength": 20.0,
            "uniform_blend": 1.0,
        }
        predicted = _fit_predict(
            "rolling_bayesian",
            parameters,
            dataset.x[:40],
            dataset.y[:40],
            dataset.x[40:41],
            seed=0,
        )
        self.assertAlmostEqual(float(predicted[0].sum()), 5.0)
        self.assertGreater(float(np.ptp(predicted[0])), 0.0)

    def test_rolling_bayesian_participates_in_nested_selection(self) -> None:
        draws = dated_random_draws(180)
        results = nested_ml_backtest(
            draws,
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            models=("rolling_bayesian",),
        )
        self.assertEqual(results[0].model, "rolling_bayesian")
        self.assertEqual(results[0].test_draws, 100)
        self.assertIn(results[0].final_parameters["window"], (10, 50, 200))

    def test_logistic_ranker_is_tuned_for_non_uniform_ranking(self) -> None:
        draws = dated_random_draws(180)
        results = nested_ml_backtest(
            draws,
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            models=("logistic_ranker",),
        )
        self.assertEqual(results[0].model, "logistic_ranker")
        self.assertEqual(results[0].final_parameters["uniform_blend"], 1.0)

    def test_ridge_ranker_uses_only_within_draw_differences(self) -> None:
        dataset = build_feature_dataset(dated_random_draws(100), min_history=20)
        parameters = {"alpha": 10.0, "uniform_blend": 1.0}
        baseline = _fit_predict(
            "ridge_ranker",
            parameters,
            dataset.x[:60],
            dataset.y[:60],
            dataset.x[60:62],
            seed=0,
        )
        common_shift = np.linspace(-5, 5, dataset.x.shape[-1])[None, None, :]
        shifted = _fit_predict(
            "ridge_ranker",
            parameters,
            dataset.x[:60] + common_shift,
            dataset.y[:60],
            dataset.x[60:62] + common_shift,
            seed=0,
        )
        np.testing.assert_allclose(baseline, shifted, atol=1e-10)
        np.testing.assert_allclose(baseline.sum(axis=1), 5.0)
        self.assertGreater(float(np.ptp(baseline)), 0.0)

    def test_ridge_ranker_participates_in_nested_selection(self) -> None:
        results = nested_ml_backtest(
            dated_random_draws(180),
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            models=("ridge_ranker",),
        )
        self.assertEqual(results[0].model, "ridge_ranker")
        self.assertIn(results[0].final_parameters["alpha"], (0.1, 1.0, 10.0, 100.0, 1000.0))
        self.assertEqual(results[0].final_parameters["uniform_blend"], 1.0)

    def test_rolling_ridge_ignores_non_rolling_features(self) -> None:
        dataset = build_feature_dataset(dated_random_draws(100), min_history=20)
        parameters = {"alpha": 100.0, "uniform_blend": 1.0}
        baseline = _fit_predict(
            "rolling_ridge_ranker",
            parameters,
            dataset.x[:60],
            dataset.y[:60],
            dataset.x[60:62],
            seed=0,
        )
        altered = dataset.x.copy()
        rolling = {
            dataset.feature_names.index(f"frequency_{window}_delta")
            for window in (10, 50, 200)
        }
        non_rolling = [index for index in range(altered.shape[-1]) if index not in rolling]
        altered[:, :, non_rolling] += 1_000
        actual = _fit_predict(
            "rolling_ridge_ranker",
            parameters,
            altered[:60],
            dataset.y[:60],
            altered[60:62],
            seed=0,
        )
        np.testing.assert_allclose(baseline, actual)

    def test_rolling_ridge_participates_in_nested_selection(self) -> None:
        draws = dated_random_draws(180)
        results = nested_ml_backtest(
            draws,
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            models=("rolling_ridge_ranker",),
        )
        self.assertEqual(results[0].model, "rolling_ridge_ranker")
        self.assertIn(results[0].final_parameters["alpha"], (0.1, 1.0, 10.0, 100.0, 1000.0))
        self.assertEqual(results[0].final_parameters["uniform_blend"], 1.0)
        first = predict_next_draw(
            draws, results, date(2021, 1, 1), force=True, seed=42
        )
        second = predict_next_draw(
            draws, results, date(2021, 1, 1), force=True, seed=42
        )
        self.assertEqual(first["numbers"], second["numbers"])
        self.assertEqual(first["chance"], second["chance"])

    def test_ranking_metric_does_not_depend_on_the_seed(self) -> None:
        draws = dated_random_draws(180)
        common = {
            "min_history": 20,
            "min_train": 60,
            "outer_folds": 2,
            "simulations": 100,
            "models": ("rolling_ridge_ranker",),
        }
        first = nested_ml_backtest(draws, seed=0, **common)[0]
        second = nested_ml_backtest(draws, seed=123, **common)[0]
        self.assertEqual(first.mean_top5_hits, second.mean_top5_hits)

    def test_marginal_probabilities_cover_all_numbers_and_sum_to_five(self) -> None:
        draws = dated_random_draws(180)
        results = nested_ml_backtest(
            draws,
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            models=("ridge_ranker",),
        )
        prediction = predict_next_draw(
            draws, results, date(2021, 1, 1), force=True, seed=0
        )
        marginals = prediction["marginal_probabilities"]
        self.assertEqual(sorted(marginals), list(range(1, 50)))
        self.assertAlmostEqual(sum(marginals.values()), 5.0, places=9)
        self.assertAlmostEqual(prediction["marginal_probability_sum"], 5.0, places=9)

    def test_forced_prediction_abstains_on_the_chance_number(self) -> None:
        draws = dated_random_draws(180)
        results = nested_ml_backtest(
            draws,
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            models=("ridge_ranker",),
        )
        prediction = predict_next_draw(
            draws, results, date(2021, 1, 1), force=True, seed=0
        )
        self.assertIsNone(prediction["chance"])
        self.assertEqual(prediction["chance_status"], "abstention")
        published = prediction["chance_prediction"]["probabilities"]
        self.assertEqual(len(published), 10)
        for probability in published.values():
            self.assertEqual(probability, 0.1)
        self.assertIsNotNone(
            prediction["chance_prediction"]["experimental"]["number"]
        )

    def test_backtest_as_of_matches_physically_truncated_history(self) -> None:
        draws = dated_random_draws(190)
        cutoff = draws[169].draw_date
        common = {
            "min_history": 20,
            "min_train": 60,
            "outer_folds": 2,
            "simulations": 100,
            "models": ("bayesian",),
        }
        frozen = nested_ml_backtest(draws, as_of=cutoff, **common)
        truncated = nested_ml_backtest(draws[:170], **common)
        self.assertEqual(frozen, truncated)

    def test_model_results_do_not_depend_on_requested_order(self) -> None:
        draws = dated_random_draws(180)
        common = {
            "min_history": 20,
            "min_train": 60,
            "outer_folds": 2,
            "simulations": 100,
        }
        forward = nested_ml_backtest(
            draws, models=("bayesian", "rolling_bayesian"), **common
        )
        reverse = nested_ml_backtest(
            draws, models=("rolling_bayesian", "bayesian"), **common
        )
        forward_by_model = {result.model: result for result in forward}
        reverse_by_model = {result.model: result for result in reverse}
        for model in forward_by_model:
            self.assertEqual(
                forward_by_model[model].mean_brier_delta,
                reverse_by_model[model].mean_brier_delta,
            )
            self.assertEqual(
                forward_by_model[model].top5_hit_uplift,
                reverse_by_model[model].top5_hit_uplift,
            )

    def test_backtest_can_isolate_the_target_game(self) -> None:
        draws = dated_random_draws(240)
        mixed = [
            Draw(
                draw.main,
                draw.chance,
                draw.draw_date,
                "loto" if index % 2 == 0 else "super_loto",
            )
            for index, draw in enumerate(draws)
        ]
        results = nested_ml_backtest(
            mixed,
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            game="super_loto",
            models=("bayesian",),
        )
        self.assertEqual(results[0].test_draws, 40)

    def test_prediction_rejects_a_date_inside_training_history(self) -> None:
        draws = dated_random_draws(180)
        results = nested_ml_backtest(
            draws,
            min_history=20,
            min_train=60,
            outer_folds=2,
            simulations=100,
            models=("bayesian",),
        )
        with self.assertRaisesRegex(ValueError, "posterieure"):
            predict_next_draw(draws, results, draws[-1].draw_date, force=True)

    def test_backtest_detects_a_synthetic_predictable_sequence(self) -> None:
        results = nested_ml_backtest(
            biased_draws(300),
            min_history=20,
            min_train=100,
            outer_folds=2,
            simulations=300,
            models=("bayesian",),
        )
        result = results[0]
        self.assertGreater(result.mean_top5_hits, 3.0)
        self.assertGreater(result.top5_ci_low, 0)
        self.assertLess(result.mean_brier_delta, 0)
        self.assertLess(result.delta_ci_high, 0)
        self.assertTrue(result.qualified)

    def test_backtest_abstains_on_an_iid_uniform_sequence(self) -> None:
        draws = dated_random_draws(400, seed=99)
        results = nested_ml_backtest(
            draws,
            min_history=20,
            min_train=150,
            outer_folds=2,
            simulations=300,
        )
        for result in results:
            self.assertFalse(
                result.qualified,
                f"{result.model} pretend battre l'uniforme sur une sequence IID",
            )
        prediction = predict_next_draw(draws, results, date(2022, 1, 1))
        self.assertEqual(prediction["status"], "abstention")

    def test_next_features_ignore_draws_on_or_after_target(self) -> None:
        draws = dated_random_draws(80)
        target = draws[40].draw_date
        expected = next_feature_matrix(draws[:40], target)
        actual = next_feature_matrix(draws, target)
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
