from dataclasses import dataclass
import numpy as np
from vbt.models.epileptor_stim_cohort import StimEpileptor
from vbt.network.connectome import difference_coupling
from vbt.simulation.integrators import ColouredAdditiveNoise,heun_stochastic_step

@dataclass(frozen=True)
class StimulatedResult:
    time: np.ndarray; state: np.ndarray
    @property
    def source(self): return self.state[:,0]-self.state[:,3]

def simulate(model,initial_state,weights,coupling_factor,noise_coeffs,dt,period,waveform,spatial_weights,seed=0):
    state=np.asarray(initial_state,float).reshape(7,-1); every=max(1,round(period/dt)); out=[]; times=[]; block=[]
    noise=ColouredAdditiveNoise(noise_coeffs,1.,seed)
    for k,scalar in enumerate(waveform):
        def rhs(y): return model.derivative(y,difference_coupling(y[0],weights,coupling_factor),scalar*spatial_weights)
        state=heun_stochastic_step(state,dt,rhs,noise.sample(dt,state.shape[1]))
        if not np.isfinite(state).all(): raise FloatingPointError(f"stim nonfinite step {k+1}")
        block.append(state.copy())
        if (k+1)%every==0: out.append(np.mean(block,axis=0)); times.append((k+1)*dt); block=[]
    return StimulatedResult(np.asarray(times),np.asarray(out))
