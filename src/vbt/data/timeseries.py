"""Load VEP source NPZ and BrainVision multiplexed float32 recordings."""
from pathlib import Path
import re
import numpy as np

def load_source(path):
    with np.load(path, allow_pickle=False) as data:
        time=np.asarray(data["time_steps"],float).reshape(-1)
        source=np.asarray(data["source_signal"],float)
    if source.ndim!=2 or source.shape[0]!=time.size or source.shape[1]!=162 or not np.isfinite(source).all():
        raise ValueError(f"invalid source timeseries {path}: {source.shape}")
    return time,source

def read_brainvision(vhdr):
    path=Path(vhdr); text=path.read_text(encoding="utf-8-sig",errors="replace")
    values=dict(line.split("=",1) for line in text.splitlines() if "=" in line and not line.startswith(";"))
    n=int(values["NumberOfChannels"]); sfreq=1e6/float(values["SamplingInterval"])
    eeg=path.with_name(values["DataFile"]); raw=np.fromfile(eeg,dtype="<f4")
    if raw.size%n: raise ValueError(f"sample count not divisible by {n}")
    names=[]
    for line in text.splitlines():
        if re.match(r"Ch\d+=",line): names.append(line.split("=",1)[1].split(",",1)[0])
    return raw.reshape(-1,n),sfreq,tuple(names)
