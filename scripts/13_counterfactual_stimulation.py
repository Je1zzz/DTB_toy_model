#!/usr/bin/env python
"""Matched-dose location-only counterfactual stimulation benchmark."""

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.data.parameters import load_epileptor_parameters,load_simulator_parameters,load_stimulation_parameters
from vbt.data.vep import VEPSubject
from vbt.evaluation.ezn import evaluate
from vbt.stimulation.waveform import biphasic_waveform

DATA=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0")

def main():
 p=argparse.ArgumentParser(); p.add_argument("--random-actions",type=int,default=20); p.add_argument("--post-window",type=float,default=500.); p.add_argument("--subjects",nargs="*"); a=p.parse_args(); rng=np.random.default_rng(0); overall=time.perf_counter()
 manifest=pd.read_csv(ROOT/"outputs/stimulation_existing/stimulation_manifest.csv");
 if a.subjects: manifest=manifest[manifest.subject.isin(a.subjects)]
 rows=[]
 for rec in manifest.to_dict("records"):
  started=time.perf_counter(); subject=rec["subject"]; twin=VEPSubject.load(DATA,subject); recording=next(x for x in twin.recordings if str(x.path)==rec["primary_recording"]); base=Path(rec["parameters"]).parent
  epi=load_epileptor_parameters(base/f"{subject}_epileptor_parameters_run-{rec['primary_recording'].split('_run-')[1].split('_')[0]}.tsv"); sim=load_simulator_parameters(next(base.glob("*simulator*run-*.tsv"))); stim=load_stimulation_parameters(rec["parameters"]); weights=twin.connectome.cohort_weights; row_sum=weights.sum(1); dt=sim.dt
  gain=np.abs(twin.bipolar_gain(recording)); gain/=np.maximum(gain.max(1,keepdims=True),np.finfo(float).tiny); truth=np.asarray([x in set(epi.ez) for x in twin.region_names]); engagement=(gain[:,truth].sum(1)/np.maximum(gain.sum(1),np.finfo(float).tiny)); oracle=int(np.argmax(engagement)); provided=next(i for i,x in enumerate(recording.channel_names) if x in set(stim.channels)); random_idx=rng.choice(gain.shape[0],size=a.random_actions,replace=gain.shape[0]<a.random_actions)
  prediction=pd.read_csv(ROOT/"outputs/personalization_map"/subject/"prediction_map.csv").sort_values("roi_index"); inferred_top=np.argsort(prediction.map_x0.to_numpy())[-10:]; inferred=int(np.argmax(gain[:,inferred_top].sum(1)/np.maximum(gain.sum(1),np.finfo(float).tiny)))
  indices=np.r_[provided,oracle,inferred,random_idx]; names=["provided","oracle","inferred_evn"]+[f"random_{i:02d}" for i in range(a.random_actions)]; spatial=gain[indices]
  state=sim.init_cond if sim.init_cond.shape[0]==7 else np.vstack([sim.init_cond,np.zeros((1,162))]); nsig=sim.noise_coeffs[:,None]; eta=np.zeros((7,162)); decay=np.exp(-dt); scale=np.sqrt(1-decay**2)
  def deriv(y,drive):
   x1,y1,z,x2,y2,g,m=np.moveaxis(y,-2,0); coupling=sim.coupling_factor*(x1@weights.T-x1*row_sum); shape=np.where(x1<0,-x1**2+3*x1,x2+.6*(z-4)**2); h=np.heaviside(m-epi.threshold,1.); f2=np.where(x2<-.25,0.,6*(x2+.25)); dz7=np.where(z<0,-.1*z**7,0.); out=np.empty_like(y); out[...,0,:]=y1-z+epi.iext+3*drive+epi.kvf*coupling+shape*x1; out[...,1,:]=1-5*x1**2-y1; out[...,2,:]=epi.r.reshape(-1)[0]*(4*(x1-epi.x0-h)+dz7-z+epi.ks*coupling); out[...,3,:]=-y2+x2-x2**3+epi.iext2+2*g-.3*(z-3.5)+epi.kf*coupling; out[...,4,:]=(-y2+f2)/10; out[...,5,:]=-.01*(g-.1*x1); out[...,6,:]=epi.r2.reshape(-1)[0]*(-.3*m+20*np.abs(drive)+epi.kf*coupling); return out
  def step(y,drive,noise): first=deriv(y,drive); pred=y+dt*first+noise; return y+.5*dt*(first+deriv(pred,drive))+noise
  burn_steps=round(stim.onset/dt); pre=[]
  for k in range(burn_steps):
   eta=decay*eta+scale*rng.standard_normal(eta.shape); state=step(state,np.zeros(162),np.sqrt(dt)*nsig*eta)
   if k>=burn_steps-round(min(500,stim.onset)/dt) and (k+1)%max(1,round(sim.period/dt))==0: pre.append(state[0]-state[3])
  state=np.repeat(state[None,:,:],len(indices),axis=0); eta_batch=np.repeat(eta[None,:,:],len(indices),axis=0); baseline=np.median(np.abs(pre),axis=0); peaks=np.zeros((len(indices),162)); duration=a.post_window; _,wave=biphasic_waveform(stim.onset+duration,dt,stim.onset,stim.period_samples,stim.amplitude,stim.pulse_width); wave=wave[burn_steps:]
  for scalar in wave:
   common=scale*rng.standard_normal((7,162)); eta_batch=decay*eta_batch+common[None,:,:]; state=step(state,scalar*spatial,np.sqrt(dt)*nsig[None,:,:]*eta_batch); peaks=np.maximum(peaks,np.abs(state[:,0]-state[:,3]))
  response=peaks-baseline; elapsed=time.perf_counter()-started
  for name,index,score in zip(names,indices,response):
   metric=evaluate(score,truth); rows.append({"subject":subject,"policy":name.split('_')[0],"action":name,"channel":recording.channel_names[index],"response_auroc":metric["auroc"],"response_ap":metric["average_precision"],"first_ez_rank":metric["first_ez_rank"],"oracle_k_recall":metric["oracle_k_recall"],"ez_target_engagement":engagement[index],"subject_forward_total_s":elapsed,"n_actions":len(indices)})
 frame=pd.DataFrame(rows); out=ROOT/"outputs/stimulation_counterfactual"; out.mkdir(parents=True,exist_ok=True); frame.to_csv(out/"action_metrics.csv",index=False); subject_policy=frame.groupby(["subject","policy"]).agg({"response_auroc":"median","ez_target_engagement":"median","subject_forward_total_s":"first"}).reset_index(); subject_policy.to_csv(out/"subject_policy_metrics.csv",index=False)
 summary={"subjects":int(frame.subject.nunique()),"actions":len(frame),"random_actions_per_subject":a.random_actions,"post_window":a.post_window,"inferred_evn_rule":"channel maximizing normalized gain engagement of top-10 MAP EZN regions; no EZ truth used for action selection","inferred_policy_gate":"EXPLORATORY_FAIL_INVERSION_QUALITY","policy_response_auroc_mean":subject_policy.groupby("policy").response_auroc.mean().to_dict(),"policy_target_engagement_mean":subject_policy.groupby("policy").ez_target_engagement.mean().to_dict(),"timing_seconds":{"wall":time.perf_counter()-overall,"per_subject_mean":float(subject_policy.groupby('subject').subject_forward_total_s.first().mean())},"scope":"synthetic diagnostic stimulation; location-only matched-dose counterfactual, not therapeutic suppression"}; (out/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
