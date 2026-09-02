#!/usr/bin/env python
"""Evaluate fitted MAP feature trajectories on later synthetic seizures."""

import csv, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.data.vep import VEPSubject
from vbt.inference.reference_engine import log_power_features, read_brainvision

DATA=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0")


def prediction(path):
    with path.open(encoding="utf-8") as handle:
        lines=(line for line in handle if not line.startswith("#")); reader=csv.reader(lines); header=next(reader); row=list(reader)[-1]
    found=[]
    for index,name in enumerate(header):
        match=re.fullmatch(r"xhat_q\.(\d+)\.(\d+)",name)
        if match: found.append((int(match.group(1)),int(match.group(2)),index))
    found.sort(); shape=(max(x[0] for x in found),max(x[1] for x in found)); values=np.asarray([float(row[x[2]]) for x in found])
    return values.reshape(shape)


def ev(pred,target):
    denominator=np.sum((target-target.mean())**2)
    return float(1-np.sum((pred-target)**2)/denominator)


def main():
    started=time.perf_counter(); rows=[]; incompatible=[]
    for subject_dir in sorted((ROOT/"outputs/personalization_map").glob("sub-*")):
        subject=subject_dir.name; report=json.loads((subject_dir/"map_report.json").read_text()); pred=prediction(subject_dir/f"optimize-{report['best_start']}.csv"); preflight=json.loads((subject_dir/"preflight.json").read_text()); fit_path=Path(preflight["selected_seizure"]); twin=VEPSubject.load(DATA,subject)
        fit_names=list(np.load(subject_dir/"prepared_inputs.npz")["channel_names"])
        candidates=[record for record in twin.recordings if record.task=="simulatedseizure" and record.acquisition=="VEPhypothesis" and record.path!=fit_path]
        for record in candidates:
            raw,sfreq,names=read_brainvision(record.path)
            common=[name for name in fit_names if name in names]
            if not common:
                incompatible.append({"subject":subject,"recording":str(record.path),"reason":"no_common_channels"}); continue
            obs=log_power_features(raw,sfreq,pred.shape[0]); target=np.log(obs)
            fit_index=[fit_names.index(name) for name in common]; heldout_index=[names.index(name) for name in common]; nt=min(pred.shape[0],target.shape[0]); pred_common=pred[:nt,fit_index]; target_common=target[:nt,heldout_index]
            rows.append({"subject":subject,"heldout_recording":str(record.path),"heldout_ev":ev(pred_common,target_common),"heldout_pearson":float(np.corrcoef(pred_common.reshape(-1),target_common.reshape(-1))[0,1]),"common_channels":len(common),"fit_channel_coverage":len(common)/len(fit_names),"heldout_channel_coverage":len(common)/len(names),"common_time_bins":nt})
    frame=pd.DataFrame(rows); out=ROOT/"outputs/heldout_seizure_ev"; out.mkdir(parents=True,exist_ok=True); frame.to_csv(out/"recording_metrics.csv",index=False)
    per_subject=frame.groupby("subject").agg(heldout_ev=("heldout_ev","mean"),heldout_pearson=("heldout_pearson","mean"),heldout_recordings=("heldout_ev","size")).reset_index() if len(frame) else pd.DataFrame(); per_subject.to_csv(out/"subject_metrics.csv",index=False)
    summary={"eligible_subjects":int(frame.subject.nunique()) if len(frame) else 0,"heldout_recordings":len(frame),"incompatible_recordings":len(incompatible),"heldout_ev_mean_subject":float(per_subject.heldout_ev.mean()) if len(per_subject) else None,"heldout_ev_median_subject":float(per_subject.heldout_ev.median()) if len(per_subject) else None,"heldout_ev_positive_subjects":int((per_subject.heldout_ev>0).sum()) if len(per_subject) else 0,"heldout_pearson_mean_subject":float(per_subject.heldout_pearson.mean()) if len(per_subject) else None,"fit_channel_coverage_mean":float(frame.fit_channel_coverage.mean()) if len(frame) else None,"contract":"fit run excluded; later VEPhypothesis synthetic seizure only; named-channel intersection; shared initial time bins; no refitting","scope":"synthetic held-out-recording generalization, not clinical validation","incompatible":incompatible,"timing_seconds":{"wall":time.perf_counter()-started}}; (out/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
