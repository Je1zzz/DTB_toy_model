import unittest
import numpy as np

from vbt.contracts import ObservationBundle, Readiness
from vbt.evaluation.ev import epileptogenicity_value
from vbt.inference.methods import METHODS, get_method
from vbt.readiness import audit_readiness


class DTBContractTests(unittest.TestCase):
    def test_observation_shape_contract(self):
        bundle = ObservationBundle("sub-001", "run-01", np.ones((3, 2)), np.arange(3) / 1000, ("A", "B"), 1000.0)
        bundle.validate()

    def test_method_menu_is_honest(self):
        self.assertEqual(get_method("map").status, "IMPLEMENTED")
        self.assertEqual(get_method("nuts").status, "IMPLEMENTED")
        with self.assertRaises(NotImplementedError): get_method("sbi")
        self.assertEqual(sum(x.status == "IMPLEMENTED" for x in METHODS.values()), 2)

    def test_ev_prefers_early_recruitment(self):
        source = -np.ones((10, 3)); source[2:, 0] = 1; source[5:, 1] = 1
        ev = epileptogenicity_value(source)
        self.assertGreater(ev[0], ev[1]); self.assertGreater(ev[1], ev[2])

    def test_surface_readiness_is_hard_gated(self):
        report = audit_readiness({"synthetic_truth", "seeg", "sc", "gain"})
        self.assertEqual(report["highest_readiness"], "cohort_synthetic")
        self.assertIn("surface", report["levels"]["patient_specific_surface"]["missing"])
