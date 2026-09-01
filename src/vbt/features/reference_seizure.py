"""Reference VEP SLP, onset and offset features."""
import numpy as np
from scipy import signal

def _filter(y,sfreq,cutoff,kind):
    b,a=signal.butter(3,2*cutoff/sfreq,kind); return signal.filtfilt(b,a,y,axis=0)
def compute_slp_sim(seeg,hpf=10.,lpf=1.,sfreq=1000.):
    y=np.asarray(seeg,float); y=_filter(y,sfreq,hpf,"highpass") if hpf else y
    window=100; padded=np.pad(y,((0,window),(0,0))); out=np.empty_like(y)
    for i in range(y.shape[0]): out[i]=np.log(np.maximum(np.mean(padded[i:i+window]**2,axis=0),np.finfo(float).tiny))
    return _filter(out,sfreq,lpf,"lowpass") if lpf else out
def compute_onset(slp,start=0,end=None,thresh=.1):
    end=slp.shape[0] if end is None else end; result=np.empty(slp.shape[1],int)
    for i in range(slp.shape[1]):
        segment=slp[start:end,i]; peak=int(np.argmax(segment)); prior=np.where(slp[start:start+peak,i]<=thresh*np.max(segment))[0]
        result[i]=start+(int(prior[-1]) if prior.size else 0)
    return result
def compute_offset(slp,start=0,end=None,thresh=.1):
    end=slp.shape[0] if end is None else end; result=np.empty(slp.shape[1],int)
    for i in range(slp.shape[1]):
        segment=slp[start:end,i]; peak=int(np.argmax(segment)); after=np.where(segment[peak:]<=thresh*np.max(segment))[0]
        result[i]=start+peak+(int(after[0]) if after.size else len(segment)-peak-1)
    return result
