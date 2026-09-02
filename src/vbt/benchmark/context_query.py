"""Frozen utilities for S1 anatomy-informed same-model synthetic episodes."""
import hashlib
import numpy as np
from vbt.models.reduced_epileptor import reference_initial_state

DEV_SUBJECTS=("sub-027","sub-008","sub-009","sub-001","sub-004","sub-030")
def select_stimulation_sites(graph,gain,count=3):
    norms=np.linalg.norm(np.asarray(gain,float),axis=0); degree=graph.processed_weights.sum(axis=1)
    candidates=np.flatnonzero((norms>=np.median(norms[norms>0]))&(degree>0))
    embedding=graph.basis[:,1:min(10,graph.rank)] if graph.rank>1 else graph.basis
    selected=[int(candidates[np.argmax(degree[candidates])])]
    while len(selected)<count:
        distances=np.min(np.linalg.norm(embedding[candidates,None,:]-embedding[np.asarray(selected)][None,:,:],axis=2),axis=1)
        for used in selected: distances[candidates==used]=-np.inf
        selected.append(int(candidates[np.argmax(distances)]))
    return tuple(selected)
def frozen_initial_state(n_regions,dt=.05): return np.repeat(reference_initial_state(dt)[:,None],n_regions,axis=1)
def impulse(site,n_regions,steps=2000,onset=200,amplitude=30.):
    drive=np.zeros((steps,n_regions)); drive[onset,site]=amplitude; return drive
def deterministic_seed(*parts): return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:8],"little")
