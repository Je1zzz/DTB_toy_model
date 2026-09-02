import unittest

import numpy as np

from vbt.contracts import Condition, ContextSet, Episode, ObservationBundle, PersonalizationResult, QueryEpisode
from vbt.evaluation.predictive import counterfactual_response, predict_query, trajectory_metrics
from vbt.evaluation.metrics import average_precision, meaningful_pair_concordance, rank_metrics
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
        model=ReducedEpileptor(self.oracle.x0,self.oracle.coupling,self.weights)
        target=counterfactual_response(model,self.query.condition.initial_state,
                                       self.query.condition.stimulation,self.gain)
        observation = ObservationBundle(
            "sub-001", "context-01", target, np.arange(80) * 0.05,
            ("A1-A2", "B1-B2"), 20.0,
        )
        context = ContextSet("sub-001", (Episode(observation, self.query.condition),))
        graph = GraphParameterization.from_connectome(np.full(3, -2.5), self.weights, 3)
        fitted = fit_context_map(context, self.weights, self.gain, graph,
                                 coupling=.7, noise_scale=0.01,
                                 gain_channel_names=("A1-A2", "B1-B2"),
                                 n_starts=3, max_nfev=300)
        self.assertTrue(fitted.diagnostics["success"])
        self.assertEqual(fitted.diagnostics["n_starts"], 3)
        self.assertLess(np.max(np.abs(fitted.x0 - self.oracle.x0)), 1.5e-1)
        self.assertTrue(np.all((fitted.x0 >= -3.5) & (fitted.x0 <= -1.0)))
        start=fitted.diagnostics["starts"][fitted.diagnostics["best_start"]]
        self.assertIn("data_jacobian_singular_values",start)
        self.assertIn("posterior_jacobian_singular_values",start)
        self.assertFalse(fitted.diagnostics["fit_coupling"])

    def test_context_sites_a_c_predict_unseen_site_b_better_than_population(self):
        def condition(site):
            stimulation=np.zeros((100,3)); stimulation[5:18,site]=1.0
            return Condition(np.zeros((2,3)),stimulation,{"site":site})
        def episode(site):
            current=condition(site)
            model=ReducedEpileptor(self.oracle.x0,self.oracle.coupling,self.weights)
            target=counterfactual_response(model,current.initial_state,current.stimulation,self.gain)
            obs=ObservationBundle("sub-001",f"site-{site}",target,np.arange(100)*0.05,
                                  ("A1-A2","B1-B2"),20.0)
            return Episode(obs,current)
        context=ContextSet("sub-001",(episode(0),episode(2)))
        graph=GraphParameterization.from_connectome(np.full(3,-2.5),self.weights,3)
        fitted=fit_context_map(context,self.weights,self.gain,graph,coupling=.7,
                               noise_scale=0.01,n_starts=3,max_nfev=400,
                               gain_channel_names=("A1-A2","B1-B2"))
        query_b=QueryEpisode("sub-001","site-1",condition(1))
        oracle_model=ReducedEpileptor(self.oracle.x0,.7,self.weights)
        fitted_model=ReducedEpileptor(fitted.x0,.7,self.weights)
        pop_model=ReducedEpileptor(np.full(3,-2.5),.7,self.weights)
        target=counterfactual_response(oracle_model,query_b.condition.initial_state,query_b.condition.stimulation,self.gain)
        personalized=counterfactual_response(fitted_model,query_b.condition.initial_state,query_b.condition.stimulation,self.gain)
        pop_prediction=counterfactual_response(pop_model,query_b.condition.initial_state,query_b.condition.stimulation,self.gain)
        self.assertLess(trajectory_metrics(personalized,target)["nrmse"],
                        trajectory_metrics(pop_prediction,target)["nrmse"])
        self.assertNotIn("site-1",fitted.context_recording_ids)

    def test_query_target_cannot_change_context_fit(self):
        current=self.query.condition
        model=ReducedEpileptor(self.oracle.x0,.7,self.weights)
        delta=counterfactual_response(model,current.initial_state,current.stimulation,self.gain)
        obs=ObservationBundle("sub-001","context-01",delta,np.arange(80)*.05,("A1-A2","B1-B2"),20.)
        context=ContextSet("sub-001",(Episode(obs,current),))
        graph=GraphParameterization.from_connectome(np.full(3,-2.5),self.weights,3)
        kwargs=dict(coupling=.7,noise_scale=.01,n_starts=1,max_nfev=100,
                    gain_channel_names=("A1-A2","B1-B2"))
        first=fit_context_map(context,self.weights,self.gain,graph,**kwargs)
        QueryEpisode("sub-001","query",current,obs)
        altered=ObservationBundle("sub-001","query",np.full_like(delta,999),np.arange(80)*.05,
                                  ("A1-A2","B1-B2"),20.)
        QueryEpisode("sub-001","query",current,altered)
        second=fit_context_map(context,self.weights,self.gain,graph,**kwargs)
        np.testing.assert_array_equal(first.x0,second.x0)
        self.assertEqual(first.objective,second.objective)

    def test_gain_channel_order_is_enforced(self):
        obs=ObservationBundle("sub-001","context",np.zeros((80,2)),np.arange(80)*.05,
                              ("A1-A2","B1-B2"),20.)
        context=ContextSet("sub-001",(Episode(obs,self.query.condition),))
        graph=GraphParameterization.from_connectome(np.full(3,-2.5),self.weights,3)
        with self.assertRaisesRegex(ValueError,"channel order"):
            fit_context_map(context,self.weights,self.gain,graph,gain_channel_names=("B1-B2","A1-A2"))

    def test_tie_aware_metrics_are_permutation_invariant(self):
        scores=np.array([.8,.8,.2,.2]); truth=np.array([True,False,True,False])
        permutation=np.array([1,0,3,2])
        self.assertEqual(average_precision(scores,truth),average_precision(scores[permutation],truth[permutation]))
        self.assertEqual(rank_metrics(scores,truth),rank_metrics(scores[permutation],truth[permutation]))
        self.assertAlmostEqual(meaningful_pair_concordance(np.array([1.,1.,0.]),np.array([1.,.9,0.])),5/6)

    def test_graph_inputs_must_be_finite(self):
        bad=self.weights.copy(); bad[0,1]=np.nan
        with self.assertRaisesRegex(ValueError,"finite"):
            GraphParameterization.from_connectome(np.full(3,-2.5),bad,2)

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
