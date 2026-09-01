import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vbt.data.vep import VEPSubject


ROOT = Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0")


class TestLoader(unittest.TestCase):
    def test_contract(self):
        s=VEPSubject.load(ROOT,"sub-001"); self.assertEqual(len(s.region_names),162); self.assertEqual(s.connectome.raw_weights.shape,(162,162)); self.assertEqual(s.gain.shape,(161,162)); self.assertTrue(np.isfinite(s.connectome.weights).all()); self.assertEqual(np.count_nonzero(np.diag(s.connectome.weights)),0)
    def test_bipolar(self):
        s=VEPSubject.load(ROOT,"sub-001"); r=next(x for x in s.recordings if x.task=="simulatedseizure"); self.assertEqual(s.bipolar_gain(r).shape,(r.n_channels,162))
