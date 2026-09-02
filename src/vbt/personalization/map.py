"""Spectral-prior multi-start MAP for the reduced context/query baseline."""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.optimize import least_squares
from vbt.contracts import ContextSet, PersonalizationResult
from vbt.models.reduced_epileptor import ReducedEpileptor
from vbt.observation.seeg import project_to_seeg
from vbt.personalization.parameterization import GraphParameterization


def fit_context_map(context: ContextSet, weights: np.ndarray, gain: np.ndarray,
                    parameterization: GraphParameterization, *, log_coupling_mean: float,
                    log_coupling_scale: float = 0.35, gamma: float = 1.0,
                    roi_x0_scale: float = 0.30, noise_scale: float = 1.0,
                    dt: float = 0.05, n_starts: int = 8, fold_id: str = "dev",
                    max_nfev: int = 1000) -> PersonalizationResult:
    """Fit shared x0,K from context only under a frozen Gaussian MAP model."""
    matrix=np.asarray(weights,float); channels=np.asarray(gain,float); n=matrix.shape[0]
    if matrix.shape!=(n,n): raise ValueError("weights must be square")
    if channels.ndim!=2 or channels.shape[1]!=n or not np.isfinite(channels).all():
        raise ValueError(f"gain must be finite with shape (channels, {n})")
    if parameterization.population_x0.shape!=(n,): raise ValueError("parameterization/weights mismatch")
    if min(log_coupling_scale,roi_x0_scale,noise_scale,dt)<=0 or n_starts<1: raise ValueError("invalid MAP settings")
    context.validate(n)
    for episode in context.episodes:
        obs=episode.observation
        if obs.seeg.shape[1]!=channels.shape[0]: raise ValueError("observation channels do not match gain rows")
        if not np.isclose(1/obs.sampling_frequency_hz,dt): raise ValueError("observation sampling interval does not match dt")
        stim=episode.condition.stimulation
        if stim is not None and stim.shape[0]!=obs.seeg.shape[0]: raise ValueError("stimulation/observation length mismatch")
    rank=parameterization.rank
    alpha_scale=roi_x0_scale*np.sqrt(n/rank)
    denom=max(float(parameterization.eigenvalues[-1]),1e-8)
    spectral_precision=(1+gamma*parameterization.eigenvalues/denom)/(alpha_scale**2)

    def parts(theta):
        alpha=theta[:-1]; coupling=float(np.exp(theta[-1])); x0=parameterization.expand(alpha)
        model=ReducedEpileptor(x0,coupling,matrix,dt=dt); data=[]
        for ep in context.episodes:
            source=model.simulate(ep.condition.initial_state,ep.observation.seeg.shape[0],ep.condition.stimulation)
            data.append(((project_to_seeg(source,channels)-ep.observation.seeg)/noise_scale).ravel())
        data=np.concatenate(data); prior=np.r_[np.sqrt(spectral_precision)*alpha,(theta[-1]-log_coupling_mean)/log_coupling_scale]
        return data,prior
    def residual(theta):
        data,prior=parts(theta); return np.r_[data,prior]
    seed=int.from_bytes(hashlib.sha256(f"{context.subject_id}|{fold_id}|r{rank}".encode()).digest()[:8],"little")
    rng=np.random.default_rng(seed); starts=[np.r_[np.zeros(rank),log_coupling_mean]]
    starts += [np.r_[rng.normal(0,1/np.sqrt(spectral_precision)),rng.normal(log_coupling_mean,log_coupling_scale)] for _ in range(n_starts-1)]
    lower=np.r_[np.full(rank,-2.5),np.log(0.05)]; upper=np.r_[np.full(rank,2.5),np.log(5.0)]
    records=[]
    for index,start in enumerate(starts):
        fit=least_squares(residual,np.clip(start,lower+1e-9,upper-1e-9),bounds=(lower,upper),max_nfev=max_nfev)
        data,prior=parts(fit.x); singular=np.linalg.svd(fit.jac,compute_uv=False)
        threshold=(singular[0]*max(fit.jac.shape)*np.finfo(float).eps) if singular.size else 0
        records.append({"index":index,"fit":fit,"data_loss":float(.5*data@data),"prior_loss":float(.5*prior@prior),
                        "singular_values":singular.tolist(),"effective_rank":int(np.sum(singular>threshold)),
                        "condition_number":float(singular[0]/singular[-1]) if singular.size and singular[-1]>0 else float("inf")})
    best=min(records,key=lambda x:x["data_loss"]+x["prior_loss"]); fit=best["fit"]
    all_theta=np.stack([x["fit"].x for x in records]); all_x0=np.stack([parameterization.expand(x[:-1]) for x in all_theta])
    serialized=[{"index":r["index"],"data_loss":r["data_loss"],"prior_loss":r["prior_loss"],
                 "total_objective":r["data_loss"]+r["prior_loss"],"success":bool(r["fit"].success),"nfev":int(r["fit"].nfev),
                 "x0":all_x0[i].tolist(),"coupling":float(np.exp(r["fit"].x[-1])),
                 "singular_values":r["singular_values"],"effective_rank":r["effective_rank"],"condition_number":r["condition_number"]}
                for i,r in enumerate(records)]
    boundary=np.isclose(fit.x,lower,atol=1e-5)|np.isclose(fit.x,upper,atol=1e-5)
    return PersonalizationResult(context.subject_id,parameterization.expand(fit.x[:-1]),float(np.exp(fit.x[-1])),
        "reduced_epileptor_graph_map",tuple(ep.observation.recording_id for ep in context.episodes),
        best["data_loss"]+best["prior_loss"],{"success":bool(fit.success),"best_start":best["index"],"seed":seed,
        "rank":rank,"n_starts":n_starts,"boundary_hit_fraction":float(boundary.mean()),
        "between_start_x0_std":all_x0.std(axis=0).tolist(),"starts":serialized,"parameterization":parameterization.manifest()})
