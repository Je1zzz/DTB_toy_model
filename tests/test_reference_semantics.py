import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from vbt.network.connectome import difference_coupling
from vbt.models.epileptor_cohort import epileptor_source
from vbt.observation.seeg import legacy_source_convention
from vbt.simulation.integrators import heun_step,ColouredAdditiveNoise
from vbt.stimulation.waveform import biphasic_waveform

class TestReferenceSemantics(unittest.TestCase):
 def test_difference(self):
  w=np.array([[0,2,0],[1,0,3],[0,4,0.]],float); x=np.array([1.,3.,7.])
  expected=np.array([2*(3-1),1*(1-3)+3*(7-3),4*(3-7)])
  np.testing.assert_allclose(difference_coupling(x,w,1),expected)
 def test_source(self):
  y=np.arange(24.).reshape(2,6,2); np.testing.assert_array_equal(epileptor_source(y),y[:,3]-y[:,0])
  np.testing.assert_array_equal(legacy_source_convention(y),y[:,0]-y[:,3])
 def test_heun(self):
  x=np.array([1.]); np.testing.assert_allclose(heun_step(x,.1,lambda q:q),1.105)
 def test_noise_reproducible(self):
  a=ColouredAdditiveNoise([1,2],seed=7); b=ColouredAdditiveNoise([1,2],seed=7)
  np.testing.assert_array_equal(a.sample(.05,3),b.sample(.05,3))
 def test_biphasic(self):
  t,w=biphasic_waveform(40,.05,8,8,2,1); self.assertAlmostEqual(float(w.sum()),0.,places=12); self.assertGreater(w.max(),0); self.assertLess(w.min(),0)
if __name__=="__main__": unittest.main()
