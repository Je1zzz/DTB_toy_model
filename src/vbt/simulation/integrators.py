"""Small deterministic/stochastic Heun kernels."""
import numpy as np

def heun_step(state, dt, rhs):
    first=rhs(state); predictor=state+dt*first; return state+0.5*dt*(first+rhs(predictor))

class ColouredAdditiveNoise:
    """Ornstein-Uhlenbeck compatibility noise; seed is reproducibility-only."""
    def __init__(self, nsig, ntau=1., seed=0):
        self.nsig=np.asarray(nsig,float); self.ntau=float(ntau); self.rng=np.random.default_rng(seed); self.eta=None
    def sample(self, dt, n_regions):
        if self.eta is None: self.eta=np.zeros((self.nsig.size,n_regions))
        decay=np.exp(-dt/self.ntau); scale=np.sqrt(1-decay**2)
        self.eta=decay*self.eta+scale*self.rng.standard_normal(self.eta.shape)
        return self.nsig[:,None]*self.eta

def heun_stochastic_step(state,dt,rhs,noise):
    increment=np.sqrt(dt)*noise; first=rhs(state); predictor=state+dt*first+increment
    return state+0.5*dt*(first+rhs(predictor))+increment
