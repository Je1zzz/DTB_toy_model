#!/usr/bin/env python
"""Timed 30-subject MAP personalization benchmark with LOO baselines."""

import argparse, ast, csv, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vbt.evaluation.ezn import evaluate


def truth(data, subject, labels):
    path = data / "derivatives/tvb" / subject / "ses-01/VEPhypothesis/parameters" / f"{subject}_epileptor_parameters_run-01.tsv"
    with path.open() as handle: ez = set(ast.literal_eval(next(csv.DictReader(handle, delimiter="\t"))["EZ"]))
    return np.asarray([label in ez for label in labels])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--jobs",type=int,default=4); parser.add_argument("--opt-starts",type=int,default=2); parser.add_argument("--opt-iter",type=int,default=50); parser.add_argument("--reuse",action="store_true"); args=parser.parse_args()
    data=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0/data/VirtualEpilepticCohort")
    out=ROOT/"outputs/personalization_map"; out.mkdir(parents=True,exist_ok=True)
    subjects=[f"sub-{i:03d}" for i in range(1,31)]
    env=os.environ.copy(); env["PYTHONPATH"]=str(ROOT/"src")
    def run(subject):
        target=out/subject; report=target/"map_report.json"
        if args.reuse and report.exists(): return subject,0
        command=[sys.executable,"-m","vbt.inference.reference_engine","--blind-only","--map-only","--subject",subject,"--output",str(target),"--opt-starts",str(args.opt_starts),"--best-inits","2","--opt-iter",str(args.opt_iter),"--max-parallel","1"]
        start=time.perf_counter(); result=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT,env=env); return subject,time.perf_counter()-start if result.returncode==0 else -result.returncode
    cohort_start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool: run_times=dict(pool.map(run,subjects))
    scores=[]; truths=[]; labels=None; reports={}
    for subject in subjects:
        frame=pd.read_csv(out/subject/"prediction_map.csv").sort_values("roi_index")
        labels=frame.roi_name.tolist(); scores.append(frame.map_x0.to_numpy()); truths.append(truth(data,subject,labels)); reports[subject]=json.loads((out/subject/"map_report.json").read_text())
    scores=np.asarray(scores); truths=np.asarray(truths); rows=[]
    for i,subject in enumerate(subjects):
        methods={"personalized_map":scores[i],"loo_mean_map":np.delete(scores,i,0).mean(0),"loo_ez_frequency":np.delete(truths,i,0).mean(0)}
        for name,value in methods.items():
            metric=evaluate(value,truths[i]); metric.update({"subject":subject,"method":name}); rows.append(metric)
    frame=pd.DataFrame(rows); frame.to_csv(out/"subject_metrics.csv",index=False)
    summary=[]
    for method,part in frame.groupby("method"):
        summary.append({"method":method,**{f"{col}_mean":float(part[col].mean()) for col in ["auroc","average_precision","first_ez_rank","oracle_k_recall"]}})
    rng=np.random.default_rng(0); paired=[]; wide=frame.pivot(index="subject",columns="method",values="auroc")
    for comparator in ("loo_mean_map","loo_ez_frequency"):
        delta=(wide["personalized_map"]-wide[comparator]).to_numpy(); signs=rng.choice((-1,1),size=(10000,delta.size)); null=np.mean(signs*delta,axis=1); boots=np.mean(rng.choice(delta,size=(10000,delta.size),replace=True),axis=1)
        paired.append({"comparison":f"personalized_map-minus-{comparator}","mean_delta_auroc":float(delta.mean()),"median_delta_auroc":float(np.median(delta)),"bootstrap_95ci":[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],"sign_flip_p_two_sided":float((1+np.sum(np.abs(null)>=abs(delta.mean())))/10001)})
    timing=[reports[s]["timing_seconds"] for s in subjects]
    report_times=[(out/s/"map_report.json").stat().st_mtime for s in subjects]; artifact_span=max(report_times)-min(report_times); median_total=float(np.median([x["total"] for x in timing]))
    result={"subjects":30,"config":{"jobs":args.jobs,"opt_starts":args.opt_starts,"opt_iter":args.opt_iter},"metrics":summary,"paired_statistics":paired,"timing_seconds":{"cohort_wall_current_command":time.perf_counter()-cohort_start,"fresh_parallel_wall_estimate_from_artifacts":artifact_span+median_total,"artifact_completion_span":artifact_span,"per_subject_total_mean":float(np.mean([x["total"] for x in timing])),"per_subject_total_median":median_total,"data_features_mean":float(np.mean([x["data_and_features"] for x in timing])),"optimization_mean":float(np.mean([x["optimization"] for x in timing]))},"claim":"synthetic cohort MAP benchmark; not clinical validation"}
    (out/"summary.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
