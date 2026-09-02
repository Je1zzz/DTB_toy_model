#!/usr/bin/env python
"""LOSO graph-basis ceiling before any SEEG inversion."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.data.vep import VEPSubject
from vbt.evaluation.metrics import average_precision,rank_metrics
from vbt.personalization.parameterization import GraphParameterization
DATA=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0")

def main():
 p=argparse.ArgumentParser(); p.add_argument("--ranks",nargs="+",type=int,default=[10,20,40]); p.add_argument("--output",type=Path,default=ROOT/"outputs/context_query/representation_ceiling"); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 subjects=[VEPSubject.load(DATA,s) for s in VEPSubject.available_subjects(DATA)]; rows=[]
 for rank in a.ranks:
  for i,subject in enumerate(subjects):
   mean=np.mean([x.model_parameters.x0 for j,x in enumerate(subjects) if j!=i],axis=0)
   graph=GraphParameterization.from_connectome(mean,subject.connectome.raw_weights,rank)
   projected=graph.expand(graph.project(subject.model_parameters.x0)); truth=np.array([name in set(subject.ez_truth) for name in subject.region_names])
   metrics=rank_metrics(projected,truth); rows.append({"subject":subject.subject_id,"rank":rank,"x0_spearman":float(spearmanr(subject.model_parameters.x0,projected).statistic),"ez_auprc":average_precision(projected,truth),"recall_at_ez_count":metrics["oracle_k_recall"],"first_ez_rank":metrics["first_ez_rank"],"ez_count":int(truth.sum()),**graph.manifest()})
 fields=list(rows[0]);
 with (a.output/"subject_metrics.csv").open("w",newline="") as f: w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
 summary=[]
 for rank in a.ranks:
  part=[r for r in rows if r["rank"]==rank]; item={"rank":rank,"subjects":len(part),"median_x0_spearman":float(np.median([r["x0_spearman"] for r in part])),"macro_auprc":float(np.mean([r["ez_auprc"] for r in part])),"mean_recall_at_ez_count":float(np.mean([r["recall_at_ez_count"] for r in part]))}; item["pass_ezn_ceiling"]=item["median_x0_spearman"]>=.90 and item["macro_auprc"]>=.90 and item["mean_recall_at_ez_count"]>=.90; summary.append(item)
 report={"benchmark":"LOSO graph-only EZN representation ceiling","uses_seeg":False,"uses_query":False,"scope":"representation only; not inversion or clinical validation","thresholds":{"median_x0_spearman":.90,"macro_auprc":.90,"mean_recall_at_ez_count":.90},"results":summary,"primary_rank_stop":not next(x for x in summary if x["rank"]==20)["pass_ezn_ceiling"]}
 payload=json.dumps(report,indent=2,sort_keys=True); (a.output/"summary.json").write_text(payload); (a.output/"summary.sha256").write_text(hashlib.sha256(payload.encode()).hexdigest()+"  summary.json\n"); print(payload)
if __name__=="__main__": main()
