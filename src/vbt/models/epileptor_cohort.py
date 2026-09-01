"""NumPy compatibility implementation of TVB's six-state VEP Epileptor."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class CohortEpileptor:
    x0: np.ndarray; iext: np.ndarray; iext2: np.ndarray; r: np.ndarray
    ks: np.ndarray; kf: np.ndarray; kvf: np.ndarray; slope: np.ndarray
    a: float=1.; b: float=3.; c: float=1.; d: float=5.; aa: float=6.; bb: float=2.; tau: float=10.; tt: float=1.; modification: bool=False

    def derivative(self, y: np.ndarray, coupling_x1: np.ndarray, coupling_x2: np.ndarray) -> np.ndarray:
        x1,y1,z,x2,y2,g=np.asarray(y,float)
        f1=np.where(x1<0, self.a*x1**3-self.b*x1**2, -(self.slope-x2+0.6*(z-4.)**2)*x1)
        h=np.where(z<0, -0.1*z**7, 0.)
        if self.modification: h=self.x0+3./(1.+np.exp(-(x1+0.5)/0.1))
        else: h=4.*(x1-self.x0)+h
        f2=np.where(x2 < -0.25, 0., self.aa*(x2+0.25))
        out=np.empty_like(y,float)
        out[0]=self.tt*(y1-f1-z+self.iext+self.kvf*coupling_x1)
        out[1]=self.tt*(self.c-self.d*x1**2-y1)
        out[2]=self.tt*self.r*(h-z+self.ks*coupling_x1)
        out[3]=self.tt*(-y2+x2-x2**3+self.iext2+self.bb*g-0.3*(z-3.5)+self.kf*coupling_x2)
        out[4]=self.tt*(-y2+f2)/self.tau
        out[5]=self.tt*(-0.01*(g-0.1*x1))
        return out

def epileptor_source(states: np.ndarray) -> np.ndarray:
    values=np.asarray(states)
    return values[...,3,:]-values[...,0,:]
