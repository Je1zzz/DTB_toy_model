#!/usr/bin/env python
"""Subject-level statistics for the frozen location-only benchmark."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs/stimulation_counterfactual"; frame=pd.read_csv(OUT/"action_metrics.csv"); rng=np.random.default_rng(0)
subjects=[]
for subject,part in frame.groupby("subject"):
    random=part[part.policy=="random"].response_auroc; row={"subject":subject,"random_median":random.median(),"random_p95":random.quantile(.95)}
    for policy in ("provided","oracle","inferred"): row[policy]=float(part[part.policy==policy].response_auroc.iloc[0])
    subjects.append(row)
subject=pd.DataFrame(subjects); subject.to_csv(OUT/"paired_subject_metrics.csv",index=False)
comparisons=[]
for policy in ("oracle","provided","inferred"):
    delta=(subject[policy]-subject.random_median).to_numpy(); n=delta.size; means=[]
    for mask in range(2**n): means.append(np.mean(delta*np.asarray([1 if (mask>>j)&1 else -1 for j in range(n)])))
    boot=np.mean(rng.choice(delta,size=(10000,n),replace=True),axis=1)
    comparisons.append({"comparison":f"{policy}-minus-random_median","mean_delta_auroc":float(delta.mean()),"median_delta_auroc":float(np.median(delta)),"bootstrap_95ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"exact_sign_flip_p_two_sided":float(np.mean(np.abs(means)>=abs(delta.mean()))),"subjects_better":int(np.sum(delta>0))})
summary={"subjects":len(subject),"comparisons":comparisons,"interpretation":"oracle tests whether location matters; provided is a dataset protocol comparator, not a clinical decision; inferred-EVN is exploratory and fails the inversion-quality gate"}; (OUT/"statistics.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
