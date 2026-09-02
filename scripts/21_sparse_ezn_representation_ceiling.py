#!/usr/bin/env python
"""Frozen graph+sparse LOSO EZN representation ceiling."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.data.vep import VEPSubject
from vbt.evaluation.metrics import average_precision,meaningful_pair_concordance,rank_metrics
from vbt.personalization.parameterization import GraphParameterization
DATA=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0"); OUT=ROOT/"outputs/context_query/sparse_representation_ceiling"
SIGMA=.05; B=.15; EPS=1e-3; RANK=20

def fit(graph,truth):
 precision=graph.spectral_precision(); projector=np.eye(truth.size)-graph.basis@graph.basis.T
 def objective(x):
  alpha,delta=graph.decompose(x); return .5*np.sum(((x-truth)/SIGMA)**2)+.5*np.sum(precision*alpha**2)+np.sum(np.sqrt(delta**2+EPS**2)-EPS)/B
 def gradient(x):
  alpha,delta=graph.decompose(x); return (x-truth)/SIGMA**2+graph.basis@(precision*alpha)+projector@(delta/(B*np.sqrt(delta**2+EPS**2)))
 result=minimize(objective,np.clip(graph.population_x0,-3.5,-1.0),jac=gradient,bounds=[(-3.5,-1.0)]*truth.size,method="L-BFGS-B",options={"maxiter":1000,"ftol":1e-10,"gtol":1e-6})
 return result
def main():
 OUT.mkdir(parents=True,exist_ok=True); subjects=[VEPSubject.load(DATA,s) for s in VEPSubject.available_subjects(DATA)]; rows=[]
 for i,s in enumerate(subjects):
  mean=np.mean([o.model_parameters.x0 for j,o in enumerate(subjects) if j!=i],axis=0); graph=GraphParameterization.from_connectome(mean,s.connectome.raw_weights,RANK); fit_result=fit(graph,s.model_parameters.x0); estimate=fit_result.x; alpha,delta=graph.decompose(estimate); truth=np.array([n in set(s.ez_truth) for n in s.region_names]); rmse=float(np.sqrt(np.mean((estimate-s.model_parameters.x0)**2))); ranks=rank_metrics(estimate,truth); effective=float((np.linalg.norm(delta,1)/np.linalg.norm(delta))**2) if np.any(delta) else 0
  rows.append({"subject":s.subject_id,"success":bool(fit_result.success),"objective":float(fit_result.fun),"niter":int(fit_result.nit),"x0_rmse":rmse,"x0_spearman":float(spearmanr(s.model_parameters.x0,estimate).statistic),"meaningful_pair_concordance_0_05":meaningful_pair_concordance(estimate,s.model_parameters.x0),"ez_auprc_tie_grouped":average_precision(estimate,truth),"recall_at_ez_count_fractional":ranks["oracle_k_recall"],"first_ez_rank_tie_aware":ranks["first_ez_rank"],"effective_sparse_support":effective})
 with (OUT/"subject_metrics_v1_1.csv").open("w",newline="") as f: w=csv.DictWriter(f,rows[0],lineterminator="\n"); w.writeheader(); w.writerows(rows)
 concordance=np.asarray([r["meaningful_pair_concordance_0_05"] for r in rows]); report={"benchmark":"LOSO graph+sparse EZN representation ceiling v1.1 tie-aware reaudit","preserves_v1_failure":True,"uses_seeg":False,"rank":RANK,"sigma_repr":SIGMA,"sparse_scale":B,"epsilon":EPS,"subjects":len(rows),"success_rate":float(np.mean([r["success"] for r in rows])),"median_x0_rmse":float(np.median([r["x0_rmse"] for r in rows])),"median_x0_spearman_report_only":float(np.median([r["x0_spearman"] for r in rows])),"median_meaningful_pair_concordance_0_05":float(np.median(concordance)),"subjects_concordance_ge_0_90":int(np.sum(concordance>=.90)),"tie_grouped_macro_auprc":float(np.mean([r["ez_auprc_tie_grouped"] for r in rows])),"mean_fractional_recall_at_ez_count":float(np.mean([r["recall_at_ez_count_fractional"] for r in rows]))}; report["pass"]=report["median_x0_rmse"]<=.10 and report["median_meaningful_pair_concordance_0_05"]>=.95 and report["subjects_concordance_ge_0_90"]>=27 and report["tie_grouped_macro_auprc"]>=.90 and report["mean_fractional_recall_at_ez_count"]>=.90; (OUT/"summary_v1_1.json").write_text(json.dumps(report,indent=2,sort_keys=True)); print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__": main()
