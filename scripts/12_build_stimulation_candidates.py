#!/usr/bin/env python
"""Build matched-dose stimulation-location policies without claiming response optimization."""

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.data.parameters import load_stimulation_parameters
from vbt.data.vep import VEPSubject

DATA=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0")


def main():
    start=time.perf_counter(); existing=pd.read_csv(ROOT/"outputs/stimulation_existing/stimulation_manifest.csv"); rows=[]; actions=[]; rng=np.random.default_rng(0)
    for record in existing.to_dict("records"):
        subject=record["subject"]; twin=VEPSubject.load(DATA,subject); recording=next(x for x in twin.recordings if str(x.path)==record["primary_recording"]); gain=np.abs(twin.bipolar_gain(recording)); gain/=np.maximum(gain.sum(axis=1,keepdims=True),np.finfo(float).tiny)
        truth=np.asarray([name in set(twin.ez_truth) for name in twin.region_names]); engagement=gain[:,truth].sum(axis=1); oracle=int(np.argmax(engagement)); stim=load_stimulation_parameters(record["parameters"])
        provided_names=set(stim.channels); provided=next((i for i,name in enumerate(recording.channel_names) if name in provided_names),None)
        random_indices=rng.choice(gain.shape[0],size=20,replace=gain.shape[0]<20); random_values=engagement[random_indices]
        rows.append({"subject":subject,"n_candidates":gain.shape[0],"provided_found":provided is not None,"provided_engagement":float(engagement[provided]) if provided is not None else np.nan,"oracle_engagement":float(engagement[oracle]),"random_median_engagement":float(np.median(random_values)),"random_p95_engagement":float(np.quantile(random_values,.95)),"oracle_minus_random_median":float(engagement[oracle]-np.median(random_values)),"provided_minus_random_median":float(engagement[provided]-np.median(random_values)) if provided is not None else np.nan,"oracle_channel":recording.channel_names[oracle],"provided_channel":recording.channel_names[provided] if provided is not None else "UNMATCHED"})
        for i,name in enumerate(recording.channel_names): actions.append({"subject":subject,"channel":name,"ez_target_engagement":float(engagement[i]),"is_oracle":i==oracle,"is_provided":i==provided,"dose_source":record["parameters"]})
    frame=pd.DataFrame(rows); frame.to_csv(ROOT/"outputs/stimulation_existing/location_policy_geometry.csv",index=False); pd.DataFrame(actions).to_csv(ROOT/"outputs/stimulation_existing/candidate_actions.csv",index=False)
    delta=frame.oracle_minus_random_median.to_numpy(); signs=np.asarray([[1 if (mask>>j)&1 else -1 for j in range(delta.size)] for mask in range(2**delta.size)]); p=float(np.mean(np.abs(np.mean(signs*delta,axis=1))>=abs(delta.mean()))); boot=np.mean(rng.choice(delta,size=(10000,delta.size),replace=True),axis=1)
    summary={"subjects":len(frame),"candidate_actions":len(actions),"provided_found":int(frame.provided_found.sum()),"oracle_ez_engagement_mean":float(frame.oracle_engagement.mean()),"provided_ez_engagement_mean":float(frame.provided_engagement.mean()),"random_median_ez_engagement_mean":float(frame.random_median_engagement.mean()),"oracle_minus_random_mean":float(delta.mean()),"oracle_minus_random_bootstrap_95ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"exact_sign_flip_p":p,"timing_seconds":{"total":time.perf_counter()-start},"scope":"geometry-only policy validation; counterfactual response forward simulation still required before calling this next-stimulation optimization"}
    (ROOT/"outputs/stimulation_existing/location_policy_geometry.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
