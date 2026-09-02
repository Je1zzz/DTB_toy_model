#!/usr/bin/env python
"""One authorized prospective Fisher/OED v2 information gate."""
from __future__ import annotations
import hashlib,json,shutil,sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.benchmark.context_query import DEV_SUBJECTS,frozen_initial_state,impulse,select_stimulation_sites
from vbt.data.vep import VEPSubject
from vbt.evaluation.predictive import counterfactual_response,trajectory_metrics
from vbt.models.reduced_epileptor import ReducedEpileptor
from vbt.personalization.parameterization import GraphParameterization
DATA=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0")
OUT=ROOT/"outputs/context_query/fisher_oed_v2"
def config_section(name):
 lines=(ROOT/"configs/context_query_model.yaml").read_text().splitlines(); values={}; active=False
 for line in lines:
  if line and not line.startswith(" "): active=line.rstrip()==name+":"; continue
  if active and line.startswith("  ") and not line.startswith("    "):
   key,value=line.strip().split(":",1); value=value.strip()
   if value.startswith("["):
    items=[x.strip() for x in value.strip("[]").split(",")]
    try: values[key]=[float(x) for x in items]
    except ValueError: values[key]=items
   elif value.lower() in ("true","false"): values[key]=value.lower()=="true"
   else:
    try: values[key]=float(value)
    except ValueError: values[key]=value
 return values
MODEL=config_section("model"); BASIS=config_section("basis"); OED=config_section("fisher_oed_v2")
PROTOCOL=config_section("protocol")
RANK=int(BASIS["rank_primary"]); COUPLING=float(MODEL["coupling_primary"])
H=float(OED["derivative_step"]); H_CHECK=float(OED["derivative_check_step"]); START,STOP=map(int,OED["analysis_steps"])
DT=float(MODEL["dt"]); TAU=float(MODEL["tau"]); ONSET=int(PROTOCOL["stimulation_onset_step"]); AMPLITUDE=float(PROTOCOL["stimulation_amplitude"])

def digest(array): return hashlib.sha256(np.asarray(array,dtype="<f8").tobytes()).hexdigest()
def response(graph,gain,initial,site,beta):
 alpha=beta/np.sqrt(graph.spectral_precision()); model=ReducedEpileptor(graph.expand(alpha),COUPLING,graph.processed_weights)
 return counterfactual_response(model,initial,impulse(site,graph.population_x0.size),gain)[START:STOP]
def jacobian(graph,gain,initial,site,h):
 beta=np.zeros((2*RANK,RANK))
 for k in range(RANK): beta[2*k,k]=h; beta[2*k+1,k]=-h
 alpha=beta/np.sqrt(graph.spectral_precision())[None,:]
 x0=graph.population_x0[None,:]+alpha@graph.basis.T
 stimulated=np.repeat(initial[None,:,:],2*RANK,axis=0); control=stimulated.copy()
 degree=graph.processed_weights.sum(axis=1); output=[]
 for step in range(STOP):
  for state,driven in ((stimulated,True),(control,False)):
   x=state[:,0,:]; z=state[:,1,:]; network=x@graph.processed_weights.T-degree[None,:]*x
   dx=1-x**3-2*x**2-z
   if driven and step==ONSET: dx[:,site] += AMPLITUDE
   dz=(4*(x-x0)-z-COUPLING*network)/TAU
   state += DT*np.stack((dx,dz),axis=1)
  if step>=START: output.append((stimulated[:,0,:]-control[:,0,:])@gain.T)
 delta=np.stack(output,axis=1)
 return np.stack([((delta[2*k]-delta[2*k+1])/(2*h)).ravel() for k in range(RANK)],axis=1)
def compute_site(task):
 graph,gain,initial,site,path=task
 if path.exists():
  try:
   saved=np.load(path); return site,saved["jacobian"],saved["jacobian_check"]
  except Exception: path.unlink()
 j=jacobian(graph,gain,initial,site,H); j_check=jacobian(graph,gain,initial,site,H_CHECK)
 temporary=path.with_suffix(".tmp.npz"); np.savez_compressed(temporary,jacobian=j,jacobian_check=j_check); temporary.replace(path)
 return site,j,j_check
def symmetric_distance(a,b):
 denominator=np.sqrt((np.mean(a*a)+np.mean(b*b))/2)
 return float(np.sqrt(np.mean((a-b)**2))/denominator) if denominator>0 else float("nan")
def logdet(matrix):
 sign,value=np.linalg.slogdet(matrix); return float(value) if sign>0 else float("-inf")
def main():
 OUT.mkdir(parents=True,exist_ok=True); cache=OUT/"cache"; cache.mkdir(exist_ok=True)
 subjects=[VEPSubject.load(DATA,s) for s in VEPSubject.available_subjects(DATA)]; by_id={s.subject_id:s for s in subjects}
 subject_rows=[]; selected_records=[]; pre_oracle_pass=True
 for sid in DEV_SUBJECTS:
  subject=by_id[sid]; population=np.mean([s.model_parameters.x0 for s in subjects if s.subject_id!=sid],axis=0)
  graph=GraphParameterization.from_connectome(population,subject.connectome.raw_weights,RANK)
  recording=next(r for r in subject.recordings if r.task=="simulatedseizure" and r.acquisition=="VEPhypothesis")
  gain=subject.bipolar_gain(recording); initial=frozen_initial_state(162)
  norms=np.linalg.norm(gain,axis=0); positive=norms[norms>0]; degree=graph.processed_weights.sum(axis=1)
  candidates=np.flatnonzero((degree>0)&(norms>=np.median(positive)))
  if candidates.size<int(OED["minimum_candidates"]): raise RuntimeError(f"{sid}: only {candidates.size} candidates")
  population_responses={int(site):response(graph,gain,initial,int(site),np.zeros(RANK)) for site in candidates}
  population_scale=float(np.sqrt(np.mean(np.concatenate([x.ravel() for x in population_responses.values()])**2)))
  sigma=population_scale*10**(-30/20); fishers={}; stabilities={}; hashes={}
  tasks=[(graph,gain,initial,int(site),cache/f"{sid}_site-{int(site):03d}.npz") for site in candidates]
  with Pool(processes=min(int(OED["workers"]),len(tasks))) as pool: computed=pool.map(compute_site,tasks)
  for site,j,j_check in computed:
   stability=float(np.linalg.norm(j-j_check)/(np.linalg.norm(j_check)+1e-12))
   fisher=j.T@j/(j.shape[0]*sigma**2); fishers[int(site)]=fisher; stabilities[int(site)]=stability; hashes[int(site)]=digest(fisher)
  identity=np.eye(RANK); best=None
  for triple in combinations(map(int,candidates),3):
   fold_scores=[logdet(identity+sum((fishers[s] for s in triple if s!=q),np.zeros((RANK,RANK)))) for q in triple]
   key=(min(fold_scores),tuple(-s for s in triple))
   if best is None or key>best[0]: best=(key,triple,fold_scores)
  _,chosen,fold_logdets=best
  old=select_stimulation_sites(graph,gain); old_worst=min(logdet(identity+sum((fishers[s] for s in old if s!=q),np.zeros((RANK,RANK)))) for q in old)
  fold_rows=[]
  for q,score in zip(chosen,fold_logdets):
   fc=sum((fishers[s] for s in chosen if s!=q),np.zeros((RANK,RANK))); posterior=identity+fc
   deff=float(np.trace(fc@np.linalg.inv(posterior))); condition=float(np.linalg.cond(posterior))
   fold_rows.append({"query_site":q,"context_sites":[s for s in chosen if s!=q],"logdet":score,"effective_dimension":deff,"posterior_condition":condition})
  stable=max(stabilities[s] for s in chosen)<=float(OED["derivative_relative_tolerance"])
  local_pass=stable and all(r["effective_dimension"]>=float(OED["effective_dimension_minimum"]) and r["posterior_condition"]<=float(OED["posterior_condition_maximum"]) for r in fold_rows)
  pre_oracle_pass &= local_pass
  subject_rows.append({"subject":sid,"candidate_count":int(candidates.size),"chosen_sites":list(chosen),"v2_worst_fold_logdet":min(fold_logdets),"v1_worst_fold_logdet":old_worst,"improvement":min(fold_logdets)-old_worst,"chosen_max_step_instability":max(stabilities[s] for s in chosen),"pre_oracle_subject_pass":local_pass,"folds":fold_rows,"candidate_fisher_hashes":{str(k):v for k,v in hashes.items()},"basis_manifest":graph.manifest(),"noise_scale_population_only":sigma})
  selected_records.append((sid,subject,population,graph,gain,initial,chosen))
 median_improvement=float(np.median([r["improvement"] for r in subject_rows])); pre_oracle_pass &= median_improvement>=float(OED["median_worst_fold_logdet_improvement_minimum"])
 oracle_rows=[]; pair_distances=[]
 if pre_oracle_pass:
  for sid,subject,population,graph,gain,initial,chosen in selected_records:
   oracle_model=ReducedEpileptor(subject.model_parameters.x0,COUPLING,graph.processed_weights); pop_model=ReducedEpileptor(population,COUPLING,graph.processed_weights); responses=[]
   for site in chosen:
    target=counterfactual_response(oracle_model,initial,impulse(site,162),gain)[START:STOP]; predicted=counterfactual_response(pop_model,initial,impulse(site,162),gain)[START:STOP]
    metrics=trajectory_metrics(predicted,target); oracle_rows.append({"subject":sid,"site":site,"oracle_population_nrmse":metrics["nrmse"],"oracle_population_ev_gap":1-metrics["explained_variance"]}); responses.append(target)
   pair_distances.extend(symmetric_distance(a,b) for a,b in combinations(responses,2))
  nrmse=np.asarray([r["oracle_population_nrmse"] for r in oracle_rows]); gaps=np.asarray([r["oracle_population_ev_gap"] for r in oracle_rows]); pair=np.asarray(pair_distances)
  oracle_pass=float(np.median(nrmse))>=float(OED["oracle_population_nrmse_minimum"]) and float(np.median(gaps))>=float(OED["oracle_population_ev_gap_minimum"]) and float(np.mean(nrmse>=.10))>=float(OED["selected_sites_nrmse_fraction_minimum"]) and float(np.mean(pair>=.10))>=float(OED["symmetric_condition_pair_fraction_minimum"])
 else: oracle_pass=False
 summary={"protocol":"Fisher/OED v2 authorized prospective","subjects":list(DEV_SUBJECTS),"rank":RANK,"coupling_fixed":COUPLING,"derivative_steps":[H,H_CHECK],"analysis_steps":[START,STOP],"median_worst_fold_logdet_improvement":median_improvement,"pre_oracle_pass":bool(pre_oracle_pass),"oracle_unlocked":bool(pre_oracle_pass),"oracle_information_pass":bool(oracle_pass),"trajectory_personalization_baseline":"AUTHORIZED_FOR_MAP" if oracle_pass else "FAIL_AT_INFORMATION_GATE"}
 if oracle_rows:
  summary.update({"median_oracle_population_nrmse":float(np.median([r["oracle_population_nrmse"] for r in oracle_rows])),"median_oracle_population_ev_gap":float(np.median([r["oracle_population_ev_gap"] for r in oracle_rows])),"selected_sites_nrmse_ge_0_10_fraction":float(np.mean([r["oracle_population_nrmse"]>=.10 for r in oracle_rows])),"symmetric_condition_pairs_ge_0_10_fraction":float(np.mean(np.asarray(pair_distances)>=.10))})
 (OUT/"subject_designs.json").write_text(json.dumps(subject_rows,indent=2,sort_keys=True)); (OUT/"oracle_rows.json").write_text(json.dumps(oracle_rows,indent=2,sort_keys=True)); (OUT/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)); shutil.rmtree(cache); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__": main()
