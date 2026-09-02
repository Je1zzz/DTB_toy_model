import unittest

import numpy as np

from vbt.contracts import Condition, ContextSet, Episode, ObservationBundle, PersonalizationResult, QueryEpisode
from vbt.evaluation.predictive import counterfactual_response, predict_query, trajectory_metrics
from vbt.models.reduced_epileptor import ReducedEpileptor, reference_initial_state
from vbt.personalization.map import fit_context_map
from vbt.personalization.parameterization import GraphParameterization


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
        graph = GraphParameterization.from_connectome(np.full(3, -2.5), self.weights, 3)
        fitted = fit_context_map(context, self.weights, self.gain, graph,
                                 log_coupling_mean=np.log(0.5), noise_scale=0.01,
                                 n_starts=3, max_nfev=300)
        self.assertTrue(fitted.diagnostics["success"])
        self.assertEqual(fitted.diagnostics["n_starts"], 3)
        self.assertLess(np.max(np.abs(fitted.x0 - self.oracle.x0)), 5e-2)

    def test_context_sites_a_c_predict_unseen_site_b_better_than_population(self):
        def condition(site):
            stimulation=np.zeros((100,3)); stimulation[5:18,site]=1.0
            return Condition(np.zeros((2,3)),stimulation,{"site":site})
        def episode(site):
            query=QueryEpisode("sub-001",f"site-{site}",condition(site))
            target=predict_query(self.oracle,query,self.weights,self.gain,100)
            obs=ObservationBundle("sub-001",f"site-{site}",target,np.arange(100)*0.05,
                                  ("A1-A2","B1-B2"),20.0)
            return Episode(obs,condition(site))
        context=ContextSet("sub-001",(episode(0),episode(2)))
        graph=GraphParameterization.from_connectome(np.full(3,-2.5),self.weights,3)
        fitted=fit_context_map(context,self.weights,self.gain,graph,log_coupling_mean=np.log(0.5),
                               noise_scale=0.01,n_starts=3,max_nfev=400)
        query_b=QueryEpisode("sub-001","site-1",condition(1))
        target=predict_query(self.oracle,query_b,self.weights,self.gain,100)
        personalized=predict_query(fitted,query_b,self.weights,self.gain,100)
        population=PersonalizationResult("sub-001",np.full(3,-2.5),0.5,"population",(),0.0)
        pop_prediction=predict_query(population,query_b,self.weights,self.gain,100)
        self.assertLess(trajectory_metrics(personalized,target)["nrmse"],
                        trajectory_metrics(pop_prediction,target)["nrmse"])
        self.assertNotIn("site-1",fitted.context_recording_ids)

    def test_condition_requires_explicit_region_shape(self):
        with self.assertRaisesRegex(ValueError, "initial_state"):
            Condition(np.zeros((2, 2))).validate(3)

    def test_duplicate_context_recording_and_wrong_dt_are_rejected(self):
        obs=ObservationBundle("sub-001","same",np.ones((3,2)),np.arange(3)*0.05,
                              ("A","B"),20.0)
        ep=Episode(obs,Condition(np.zeros((2,3))))
        with self.assertRaisesRegex(ValueError,"unique"):
            ContextSet("sub-001",(ep,ep)).validate(3)
        bad=ObservationBundle("sub-001","bad",np.ones((3,2)),np.array([0.0,0.05,0.11]),
                              ("A","B"),20.0)
        with self.assertRaisesRegex(ValueError,"regular"):
            bad.validate()

    def test_counterfactual_control_uses_same_candidate_parameters(self):
        init=np.repeat(reference_initial_state()[:,None],3,axis=1)
        stim=np.zeros((30,3)); stim[5,1]=30
        model=ReducedEpileptor(self.oracle.x0,0.5,self.weights)
        delta=counterfactual_response(model,init,stim,self.gain)
        manual=model.simulate(init,30,stim)@self.gain.T-model.simulate(init,30,np.zeros_like(stim))@self.gain.T
        np.testing.assert_allclose(delta,manual)


if __name__ == "__main__":
    unittest.main()
