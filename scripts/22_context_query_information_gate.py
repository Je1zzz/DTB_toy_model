#!/usr/bin/env python
"""Six-subject pre-optimization information gate for counterfactual ΔSEEG."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.benchmark.context_query import DEV_SUBJECTS,frozen_initial_state,impulse,select_stimulation_sites
from vbt.data.vep import VEPSubject
from vbt.evaluation.predictive import counterfactual_response,trajectory_metrics
from vbt.models.reduced_epileptor import ReducedEpileptor
from vbt.personalization.parameterization import GraphParameterization
DATA=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0"); OUT=ROOT/"outputs/context_query/information_gate"
def main():
 all_subjects=[VEPSubject.load(DATA,s) for s in VEPSubject.available_subjects(DATA)]; by_id={s.subject_id:s for s in all_subjects}; rows=[]
 for sid in DEV_SUBJECTS:
  subject=by_id[sid]; population=np.mean([s.model_parameters.x0 for s in all_subjects if s.subject_id!=sid],axis=0); graph=GraphParameterization.from_connectome(population,subject.connectome.raw_weights,20); recording=next(r for r in subject.recordings if r.task=="simulatedseizure" and r.acquisition=="VEPhypothesis"); gain=subject.bipolar_gain(recording); sites=select_stimulation_sites(graph,gain); init=frozen_initial_state(162); oracle_model=ReducedEpileptor(subject.model_parameters.x0,.5,graph.processed_weights); population_model=ReducedEpileptor(population,.5,graph.processed_weights); oracle=[]
  for site in sites:
   stim=impulse(site,162); target=counterfactual_response(oracle_model,init,stim,gain); pop=counterfactual_response(population_model,init,stim,gain); metrics=trajectory_metrics(pop,target); rows.append({"subject":sid,"site":site,"oracle_population_nrmse":metrics["nrmse"],"oracle_population_ev_gap":1-metrics["explained_variance"]}); oracle.append(target)
  for i in range(3):
   for j in range(i): rows.append({"subject":sid,"site_pair":f"{sites[j]}-{sites[i]}","condition_nrmse":trajectory_metrics(oracle[i],oracle[j])["nrmse"]})
 pair=[r["condition_nrmse"] for r in rows if "condition_nrmse" in r]; pred=[r for r in rows if "oracle_population_nrmse" in r]; report={"subjects":list(DEV_SUBJECTS),"coupling":.5,"endpoint":"candidate-specific counterfactual delta SEEG","median_condition_nrmse":float(np.median(pair)),"condition_pairs_ge_0_10":float(np.mean(np.asarray(pair)>=.10)),"median_oracle_population_nrmse":float(np.median([r["oracle_population_nrmse"] for r in pred])),"median_oracle_population_ev_gap":float(np.median([r["oracle_population_ev_gap"] for r in pred]))}; report["pass"]=report["condition_pairs_ge_0_10"]>=.90 and report["median_oracle_population_nrmse"]>=.10 and report["median_oracle_population_ev_gap"]>=.10; OUT.mkdir(parents=True,exist_ok=True); (OUT/"summary.json").write_text(json.dumps(report,indent=2)); (OUT/"rows.json").write_text(json.dumps(rows,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
