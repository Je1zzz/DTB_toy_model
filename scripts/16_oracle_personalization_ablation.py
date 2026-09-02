#!/usr/bin/env python
"""Oracle FULL/SC_ONLY/DYN_ONLY/POP/MISMATCHED propagation ablation."""
import argparse,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.data.vep import VEPSubject,ModelParameters,SimulatorParameters
from vbt.data.timeseries import load_source
from vbt.features.reference_seizure import compute_slp_sim,compute_onset
from vbt.simulation.cohort import simulate
DATA=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0")

def mean_model(items):
 def avg(name): return np.mean([getattr(x,name) for x in items],axis=0)
 return ModelParameters(avg("x0"),avg("i_ext"),avg("i_ext2"),avg("slope"),float(np.mean([x.r for x in items])),avg("k_s"),avg("k_f"),avg("k_vf"),(),(),Path("LOO_POPULATION"))
def mean_sim(items):
 return SimulatorParameters(float(np.mean([x.coupling_factor for x in items])),np.mean([x.noise_coeffs for x in items],axis=0),np.mean([x.initial_state for x in items],axis=0),float(np.median([x.dt for x in items])),float(np.median([x.period for x in items])),Path("LOO_POPULATION"))
def jaccard(a,b,k=20):
 x=set(np.argsort(a)[:k]); y=set(np.argsort(b)[:k]); return len(x&y)/len(x|y)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--jobs",type=int,default=4); p.add_argument("--duration",type=float,default=4500.); p.add_argument("--subjects",nargs="*"); a=p.parse_args(); total=time.perf_counter(); subjects=[f"sub-{i:03d}" for i in range(1,31)]; twins={s:VEPSubject.load(DATA,s) for s in subjects}; chosen=a.subjects or subjects; out=ROOT/"outputs/oracle_personalization"; out.mkdir(parents=True,exist_ok=True)
 def run(subject):
  target=twins[subject]; others=[twins[s] for s in subjects if s!=subject]; pop_model=mean_model([x.model_parameters for x in others]); pop_sim=mean_sim([x.simulator_parameters for x in others]); pop_sc=np.mean([x.connectome.raw_weights for x in others],axis=0); donor=twins[subjects[(subjects.index(subject)+1)%len(subjects)]]
  source_file=next(x for x in target.source_files() if "ses-01" in str(x) and "VEPhypothesis" in str(x) and "run-01" in x.name); t,source=load_source(source_file); source=source[:int(a.duration)]; target_onset=compute_onset(compute_slp_sim(source,sfreq=1000.)); arms={"FULL":(target,target.model_parameters,target.simulator_parameters,target.connectome.raw_weights),"SC_ONLY":(target,pop_model,pop_sim,target.connectome.raw_weights),"DYN_ONLY":(target,target.model_parameters,target.simulator_parameters,pop_sc),"POP":(target,pop_model,pop_sim,pop_sc),"MISMATCHED":(target,donor.model_parameters,donor.simulator_parameters,donor.connectome.raw_weights)}; rows=[]
  for name,(base,model,simulator,raw_sc) in arms.items():
   started=time.perf_counter(); conn=replace(base.connectome,raw_weights=raw_sc); proxy=replace(base,connectome=conn,model_parameters=model,simulator_parameters=simulator); result=simulate(proxy,duration=a.duration,noise=True,seed=0); onset=compute_onset(compute_slp_sim(result.source_activity,sfreq=1000.)); elapsed=time.perf_counter()-started; rho=float(spearmanr(target_onset,onset).statistic); rows.append({"subject":subject,"arm":name,"early_recruitment_jaccard_at20":jaccard(target_onset,onset),"onset_rank_spearman":rho,"forward_and_features_s":elapsed,"finite":True})
  return rows
 with ThreadPoolExecutor(max_workers=a.jobs) as pool: rows=[r for group in pool.map(run,chosen) for r in group]
 frame=pd.DataFrame(rows); frame.to_csv(out/"subject_arm_metrics.csv",index=False); summary=[]
 for arm,part in frame.groupby("arm"): summary.append({"arm":arm,"jaccard_mean":float(part.early_recruitment_jaccard_at20.mean()),"jaccard_median":float(part.early_recruitment_jaccard_at20.median()),"spearman_mean":float(part.onset_rank_spearman.mean()),"time_mean_s":float(part.forward_and_features_s.mean())})
 rng=np.random.default_rng(0); comparisons=[]; wide=frame.pivot(index="subject",columns="arm",values="early_recruitment_jaccard_at20")
 for comp in ("SC_ONLY","DYN_ONLY","POP","MISMATCHED"):
  d=(wide.FULL-wide[comp]).to_numpy(); signs=rng.choice((-1,1),size=(10000,d.size)); null=np.mean(signs*d,axis=1); boot=np.mean(rng.choice(d,size=(10000,d.size),replace=True),axis=1); comparisons.append({"comparison":f"FULL-minus-{comp}","mean_delta":float(d.mean()),"bootstrap_95ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"sign_flip_p":float((1+np.sum(np.abs(null)>=abs(d.mean())))/10001)})
 report={"subjects":len(chosen),"duration":a.duration,"endpoint":"early-recruitment Jaccard@20 against provided synthetic source; secondary onset-rank Spearman","arms":summary,"comparisons":comparisons,"timing_seconds":{"wall":time.perf_counter()-total},"scope":"oracle synthetic personalization value; original RNG/trajectory not reproduced"}; (out/"summary.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
