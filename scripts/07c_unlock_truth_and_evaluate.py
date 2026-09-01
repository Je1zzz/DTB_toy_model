#!/usr/bin/env python
import argparse,ast,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); DATA=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0/data/VirtualEpilepticCohort")
from vbt.evaluation.ezn import evaluate
parser=argparse.ArgumentParser(); parser.add_argument("--smoke",action="store_true"); args=parser.parse_args(); OUT=ROOT/("outputs/phase7_smoke" if args.smoke else "outputs/phase7")
expected=json.loads((OUT/"prediction_hashes_pre_unlock.json").read_text()); current={p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/"inference").glob("sub-*/prediction_blind.csv"))}
if current != expected: raise RuntimeError("prediction hashes changed before truth unlock")
rows=[]; hashes=dict(current)
for prediction in sorted((OUT/"inference").glob("sub-*/prediction_blind.csv")):
 subject=prediction.parent.name; frame=pd.read_csv(prediction).sort_values("roi_index"); labels=frame.roi_name.tolist(); scores=frame.posterior_mean_x0.to_numpy(); gt=DATA/"derivatives/tvb"/subject/"ses-01/VEPhypothesis/parameters"/f"{subject}_epileptor_parameters_run-01.tsv"
 with gt.open() as h: ez=set(ast.literal_eval(next(csv.DictReader(h,delimiter="\t"))["EZ"])); truth=np.array([x in ez for x in labels]); metric=evaluate(scores,truth); report=json.loads((prediction.parent/"blind_run_report.json").read_text()); metric.update({"subject":subject,"optimizer_usable":report["optimizations_usable"]>0,"posterior_usable":all(v==0 for v in report["chain_return_codes"].values()),"rhat_x0_max":report["rhat_x0_max"]}); rows.append(metric)
evaluation=OUT/"evaluation"; evaluation.mkdir(exist_ok=True); pd.DataFrame(rows).to_csv(evaluation/"unlocked_subject_metrics.csv",index=False); (evaluation/"prediction_hashes_before_unlock.json").write_text(json.dumps(hashes,indent=2)); after={p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/"inference").glob("sub-*/prediction_blind.csv"))}; unchanged=after==hashes; (evaluation/"prediction_hashes_after_evaluation.json").write_text(json.dumps(after,indent=2)); (evaluation/"leakage_audit.json").write_text(json.dumps({"prediction_hashes_unchanged":unchanged,"inference_launched_by_unlock":False,"evaluation_output_only":True},indent=2)); print(json.dumps({"evaluated":len(rows),"prediction_hashes_unchanged":unchanged},indent=2))
