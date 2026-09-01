import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from vbt.data.parameters import load_epileptor_parameters,load_simulator_parameters,load_stimulation_parameters
from vbt.evaluation.ezn import evaluate
from vbt.features.reference_seizure import compute_slp_sim,compute_onset
from vbt.models.epileptor_stim_cohort import StimEpileptor
from vbt.stimulation.spatial import reconstruct_weights
DATA=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0/data/VirtualEpilepticCohort")
PARAM=DATA/"derivatives/tvb/sub-002/ses-02/VEPhypothesis/parameters"
class TestComponents(unittest.TestCase):
 def test_parameter_roundtrip(self):
  e=load_epileptor_parameters(next(PARAM.glob("*epileptor*run-01.tsv"))); s=load_simulator_parameters(next(PARAM.glob("*simulator*run-01.tsv"))); p=load_stimulation_parameters(next(PARAM.glob("*stimulation*run-01.tsv")))
  self.assertEqual(e.x0.shape,(162,)); self.assertIn(s.init_cond.shape,( (6,162),(7,162) )); self.assertEqual(p.weights.shape,(162,)); self.assertEqual(p.channels,("B'1-2",))
 def test_stim_derivative(self):
  one=np.ones(2); model=StimEpileptor(-2.2*one,3*one,3.1*one,.45*one,.00025*one,.00025*one,-3*one,-.22*one,-.085*one); y=np.zeros((7,2)); y[2]=3
  d=model.derivative(y,np.zeros(2),np.zeros(2)); self.assertEqual(d.shape,y.shape); self.assertTrue(np.isfinite(d).all())
 def test_slp_and_onset(self):
  t=np.arange(2000)/1000; x=np.column_stack([np.sin(2*np.pi*20*t),np.sin(2*np.pi*30*t)]); slp=compute_slp_sim(x,sfreq=1000); self.assertEqual(slp.shape,x.shape); self.assertEqual(compute_onset(slp).shape,(2,))
 def test_spatial_normalization(self):
  np.testing.assert_allclose(reconstruct_weights([2,4,6]),[0,.5,1])
 def test_ezn_direction(self):
  result=evaluate(np.array([3.,2.,1.,0.]),np.array([1,1,0,0],bool)); self.assertEqual(result["auroc"],1.); self.assertEqual(result["first_ez_rank"],1)
if __name__=="__main__": unittest.main()
