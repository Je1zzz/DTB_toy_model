import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vbt.data.vep import VEPSubject
from vbt.observation.seeg import project_to_seeg
from vbt.simulation.simulator import simulate_spontaneous
ROOT = Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0")
class TestForward(unittest.TestCase):
    def test_shapes(self):
        subject=VEPSubject.load(ROOT,"sub-001"); result=simulate_spontaneous(subject,duration=2.0); source=result.source_activity
        recording=next(x for x in subject.recordings if x.task=="simulatedseizure"); seeg=project_to_seeg(source,subject.bipolar_gain(recording))
        self.assertEqual(source.shape[1],162); self.assertEqual(seeg.shape,(source.shape[0],recording.n_channels)); self.assertTrue(np.isfinite(source).all()); self.assertFalse(np.allclose(source,source[0]))
