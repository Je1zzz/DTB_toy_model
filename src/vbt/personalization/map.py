"""Spectral-prior multi-start MAP for counterfactual context responses."""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.optimize import least_squares
from vbt.contracts import ContextSet, PersonalizationResult
from vbt.evaluation.predictive import counterfactual_response
from vbt.models.reduced_epileptor import ReducedEpileptor
from vbt.personalization.parameterization import GraphParameterization


def fit_context_map(context: ContextSet, weights: np.ndarray, gain: np.ndarray,
                    parameterization: GraphParameterization, *, coupling: float = 0.5,
                    fit_coupling: bool = False, log_coupling_mean: float | None = None,
                    log_coupling_scale: float = 0.35, gamma: float = 1.0,
                    roi_x0_scale: float = 0.30, noise_scale: float = 1.0,
                    dt: float = 0.05, n_starts: int = 8, fold_id: str = "dev",
                    max_nfev: int = 1000, gain_channel_names: tuple[str, ...] | None = None,
                    x0_bounds: tuple[float, float] = (-3.5, -1.0)) -> PersonalizationResult:
    """Fit shared x0 from context ΔSEEG; coupling is fixed by default."""
    matrix=np.asarray(weights,float); channels=np.asarray(gain,float); n=matrix.shape[0]
    if matrix.shape!=(n,n) or not np.isfinite(matrix).all(): raise ValueError("weights must be finite and square")
    if channels.ndim!=2 or channels.shape[1]!=n or not np.isfinite(channels).all():
        raise ValueError(f"gain must be finite with shape (channels, {n})")
    if parameterization.population_x0.shape!=(n,): raise ValueError("parameterization/weights mismatch")
    if min(log_coupling_scale,roi_x0_scale,noise_scale,dt,coupling)<=0 or n_starts<1: raise ValueError("invalid MAP settings")
    if fit_coupling and log_coupling_mean is None: raise ValueError("log_coupling_mean is required when fit_coupling=True")
    if gain_channel_names is None or len(gain_channel_names)!=channels.shape[0] or len(set(gain_channel_names))!=len(gain_channel_names):
        raise ValueError("gain_channel_names must uniquely identify gain rows")
    context.validate(n)
    for episode in context.episodes:
        obs=episode.observation
        if tuple(obs.channel_names)!=tuple(gain_channel_names):
            raise ValueError("observation channel order does not match gain_channel_names")
        if not np.isclose(1/obs.sampling_frequency_hz,dt): raise ValueError("observation sampling interval does not match dt")
        stim=episode.condition.stimulation
        if stim is None: raise ValueError("counterfactual ΔSEEG context requires stimulation")
        if stim.shape[0]!=obs.seeg.shape[0]: raise ValueError("stimulation/observation length mismatch")
    rank=parameterization.rank
    spectral_precision=parameterization.spectral_precision(roi_x0_scale,gamma)
    lower_x0,upper_x0=x0_bounds; population=parameterization.population_x0
    if lower_x0>=upper_x0 or np.any(population<=lower_x0) or np.any(population>=upper_x0):
        raise ValueError("population_x0 must be strictly inside x0_bounds")
    row_l1=np.sum(np.abs(parameterization.basis),axis=1)
    alpha_limit=float(np.min(np.minimum(population-lower_x0,upper_x0-population)/np.maximum(row_l1,1e-12)))
    if alpha_limit<=0: raise ValueError("no feasible spectral coefficient range")

    def unpack(theta):
        alpha=theta[:rank]
        fitted_coupling=float(np.exp(theta[-1])) if fit_coupling else float(coupling)
        return alpha,fitted_coupling,parameterization.expand(alpha)
    def parts(theta):
        alpha,fitted_coupling,x0=unpack(theta); model=ReducedEpileptor(x0,fitted_coupling,matrix,dt=dt); data=[]
        for ep in context.episodes:
            predicted=counterfactual_response(model,ep.condition.initial_state,ep.condition.stimulation,channels)
            data.append(((predicted-ep.observation.seeg)/noise_scale).ravel())
        prior=list(np.sqrt(spectral_precision)*alpha)
        if fit_coupling: prior.append((theta[-1]-float(log_coupling_mean))/log_coupling_scale)
        return np.concatenate(data),np.asarray(prior)
    def residual(theta):
        data,prior=parts(theta); return np.r_[data,prior]
    seed=int.from_bytes(hashlib.sha256(f"{context.subject_id}|{fold_id}|r{rank}".encode()).digest()[:8],"little")
    rng=np.random.default_rng(seed); starts=[np.zeros(rank)]
    starts += [rng.normal(0,1/np.sqrt(spectral_precision)) for _ in range(n_starts-1)]
    lower=np.full(rank,-alpha_limit); upper=np.full(rank,alpha_limit)
    if fit_coupling:
        starts=[np.r_[s,float(log_coupling_mean) if i==0 else rng.normal(float(log_coupling_mean),log_coupling_scale)] for i,s in enumerate(starts)]
        lower=np.r_[lower,np.log(0.05)]; upper=np.r_[upper,np.log(5.0)]
    records=[]
    for index,start in enumerate(starts):
        fit=least_squares(residual,np.clip(start,lower+1e-9,upper-1e-9),bounds=(lower,upper),max_nfev=max_nfev)
        data,prior=parts(fit.x); data_jac=fit.jac[:data.size]; data_s=np.linalg.svd(data_jac,compute_uv=False)
        posterior_s=np.linalg.svd(fit.jac,compute_uv=False)
        effective_rank=int(np.sum(data_s/data_s[0]>=1e-6)) if data_s.size and data_s[0]>0 else 0
        condition=float(data_s[0]/data_s[effective_rank-1]) if effective_rank else float("inf")
        records.append({"index":index,"fit":fit,"data_loss":float(.5*data@data),"prior_loss":float(.5*prior@prior),
                        "data_singular_values":data_s.tolist(),"posterior_singular_values":posterior_s.tolist(),
                        "data_effective_rank":effective_rank,"data_condition_number":condition})
    best=min(records,key=lambda x:x["data_loss"]+x["prior_loss"]); fit=best["fit"]
    unpacked=[unpack(r["fit"].x) for r in records]; all_x0=np.stack([u[2] for u in unpacked])
    serialized=[]
    for i,r in enumerate(records):
        serialized.append({"index":r["index"],"data_loss":r["data_loss"],"prior_loss":r["prior_loss"],
            "total_objective":r["data_loss"]+r["prior_loss"],"success":bool(r["fit"].success),"nfev":int(r["fit"].nfev),
            "x0":all_x0[i].tolist(),"coupling":unpacked[i][1],"data_jacobian_singular_values":r["data_singular_values"],
            "posterior_jacobian_singular_values":r["posterior_singular_values"],"data_effective_rank":r["data_effective_rank"],
            "data_condition_number":r["data_condition_number"]})
    _,best_coupling,best_x0=unpack(fit.x)
    roi_boundary=np.isclose(best_x0,lower_x0,atol=1e-5)|np.isclose(best_x0,upper_x0,atol=1e-5)
    return PersonalizationResult(context.subject_id,best_x0,best_coupling,"reduced_epileptor_graph_delta_map",
        tuple(ep.observation.recording_id for ep in context.episodes),best["data_loss"]+best["prior_loss"],
        {"success":bool(fit.success),"best_start":best["index"],"seed":seed,"rank":rank,"n_starts":n_starts,
         "fit_coupling":fit_coupling,"alpha_feasible_limit":alpha_limit,"roi_x0_boundary_hit_fraction":float(roi_boundary.mean()),
         "between_start_x0_std":all_x0.std(axis=0).tolist(),"starts":serialized,"parameterization":parameterization.manifest()})
