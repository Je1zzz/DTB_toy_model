"""VEP cohort spontaneous replay with explicit monitor and diagnostics."""
from dataclasses import dataclass
import numpy as np
from vbt.models.epileptor_cohort import CohortEpileptor,epileptor_source
from vbt.network.connectome import difference_coupling
from vbt.simulation.integrators import ColouredAdditiveNoise,heun_step,heun_stochastic_step

@dataclass(frozen=True)
class CohortSimulation:
    time: np.ndarray; state: np.ndarray; diagnostics: tuple[dict,...]
    @property
    def source_activity(self): return epileptor_source(self.state)

def simulate(subject,duration=None,noise=True,seed=0,dt=None):
    p=subject.model_parameters; s=subject.simulator_parameters; step=float(dt or s.dt)
    duration=float(duration or 4500); every=max(1,round(s.period/step)); n=round(duration/step)
    model=CohortEpileptor(p.x0,p.i_ext,p.i_ext2,np.full(162,p.r),p.k_s,p.k_f,p.k_vf,p.slope)
    state=np.repeat(s.initial_state[:,None],162,axis=1); weights=subject.connectome.cohort_weights
    def rhs(y): return model.derivative(y,difference_coupling(y[0],weights,s.coupling_factor),difference_coupling(y[3],weights,s.coupling_factor))
    generator=ColouredAdditiveNoise(s.noise_coeffs,1.,seed); times=[]; states=[]; diag=[]; block=[]
    for k in range(n):
        old=state
        state=heun_stochastic_step(state,step,rhs,generator.sample(step,162)) if noise else heun_step(state,step,rhs)
        if not np.isfinite(state).all(): raise FloatingPointError(f"nonfinite at step {k+1}, time={(k+1)*step}")
        if (k+1)%1000==0: diag.append({"time":(k+1)*step,"state_min":float(state.min()),"state_max":float(state.max()),"max_abs":float(np.abs(state).max())})
        block.append(state.copy())
        if (k+1)%every==0:
            times.append((k+1)*step); states.append(np.mean(block,axis=0)); block=[]
    return CohortSimulation(np.asarray(times),np.asarray(states),tuple(diag))
