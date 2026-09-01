"""Auditable algebra of generator EpileptorStim2Populations (7 states).

The generator declares ``cvar=[0]`` but its numba kernel reads three coupling
components. That is an upstream shape inconsistency. This compatibility layer
uses the sole Difference-coupled x1 component for all three reads and reports
the choice; it cannot be called bitwise equation-equivalent to the undefined
out-of-bounds reference behavior.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class StimEpileptor:
    x0: np.ndarray; threshold: np.ndarray; iext: np.ndarray; iext2: np.ndarray
    r: np.ndarray; r2: np.ndarray; ks: np.ndarray; kf: np.ndarray; kvf: np.ndarray
    def derivative(self,y,coupling,istim):
        x1,y1,z,x2,y2,g,m=np.asarray(y,float); c1=np.asarray(coupling,float)
        shape=np.where(x1<0,-x1**2+3*x1,x2+0.6*(z-4)**2)
        dz7=np.where(z<0,-0.1*z**7,0.); h=np.heaviside(m-self.threshold,1.)
        f2=np.where(x2<-.25,0.,6*(x2+.25)); out=np.empty_like(y,float)
        out[0]=y1-z+self.iext+3*istim+self.kvf*c1+shape*x1
        out[1]=1-5*x1**2-y1
        out[2]=self.r*(4*(x1-self.x0-h)+dz7-z+self.ks*c1)
        out[3]=-y2+x2-x2**3+self.iext2+2*g-.3*(z-3.5)+self.kf*c1
        out[4]=(-y2+f2)/10
        out[5]=-.01*(g-.1*x1)
        out[6]=self.r2*(-.3*m+20*np.abs(istim)+self.kf*c1)
        return out
