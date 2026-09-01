#!/usr/bin/env python
import ast,csv,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.evaluation.ezn import evaluate,permutation_p
def main():
 pred=pd.read_csv(ROOT/"outputs/phase5/sub-001/prediction_blind.csv"); labels=pred.sort_values("roi_index")["roi_name"].tolist(); scores=pred.sort_values("roi_index")["posterior_mean_x0"].to_numpy(); gt=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0/data/VirtualEpilepticCohort/derivatives/tvb/sub-001/ses-01/VEPhypothesis/parameters/sub-001_epileptor_parameters_run-01.tsv")
 with gt.open() as h: row=next(csv.DictReader(h,delimiter="\t")); ez=set(ast.literal_eval(row["EZ"])); truth=np.array([x in ez for x in labels]); result=evaluate(scores,truth); result.update(permutation_p(scores,truth)); result.update({"phase":"6","subject":"sub-001","G6A":"PASS","G6B":"METRIC-SANITY PASS, INFERENCE-VALIDITY NOT PASS" if result["auroc"]>result["auroc_null_p95"] else "FAIL","scientific_EZN_inference":"NOT VALIDATED: Phase 5 posterior convergence failed"}); out=ROOT/"outputs/phase6/sub-001"; out.mkdir(parents=True,exist_ok=True); (out/"phase6_report.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
