import unittest

import numpy as np

from vbt.contracts import Condition, ContextSet, Episode, ObservationBundle, PersonalizationResult, QueryEpisode
from vbt.evaluation.predictive import predict_query, trajectory_metrics
from vbt.personalization.map import fit_context_map


class PersonalizationPredictiveTests(unittest.TestCase):
    def setUp(self):
        self.weights = np.array([[0.0, 0.4, 0.1], [0.4, 0.0, 0.2], [0.1, 0.2, 0.0]])
        self.gain = np.array([[1.0, -0.5, 0.25], [-0.2, 0.4, -0.7]])
        stimulation = np.zeros((80, 3)); stimulation[5:12, 0] = 0.8
        self.query = QueryEpisode("sub-001", "query-01", Condition(np.zeros((2, 3)), stimulation))
        self.oracle = PersonalizationResult("sub-001", np.array([-1.9, -2.2, -2.3]), 0.7, "oracle", ("context-01",), 0.0)

    def test_query_condition_is_used_to_rerun_forward(self):
        target = predict_query(self.oracle, self.query, self.weights, self.gain, 80)
        population = PersonalizationResult("sub-001", np.full(3, -2.5), 0.1, "population", (), 1.0)
        prediction = predict_query(population, self.query, self.weights, self.gain, 80)
        self.assertAlmostEqual(trajectory_metrics(target, target)["explained_variance"], 1.0)
        self.assertLess(trajectory_metrics(prediction, target)["explained_variance"], 1.0)

    def test_subject_leakage_is_rejected(self):
        wrong = QueryEpisode("sub-999", "query-01", self.query.condition)
        with self.assertRaisesRegex(ValueError, "subject"):
            predict_query(self.oracle, wrong, self.weights, self.gain, 80)

    def test_map_fits_shared_parameters_from_context(self):
        target = predict_query(self.oracle, self.query, self.weights, self.gain, 80)
        observation = ObservationBundle(
            "sub-001", "context-01", target, np.arange(80) * 0.05,
            ("A1-A2", "B1-B2"), 20.0,
        )
        context = ContextSet("sub-001", (Episode(observation, self.query.condition),))
        fitted = fit_context_map(context, self.weights, self.gain, max_nfev=200)
        self.assertTrue(fitted.diagnostics["success"])
        self.assertLess(fitted.objective, 1e-8)
        self.assertLess(np.max(np.abs(fitted.x0 - self.oracle.x0)), 1e-2)

    def test_condition_requires_explicit_region_shape(self):
        with self.assertRaisesRegex(ValueError, "initial_state"):
            Condition(np.zeros((2, 2))).validate(3)


if __name__ == "__main__":
    unittest.main()
