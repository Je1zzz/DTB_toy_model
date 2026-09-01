#!/usr/bin/env python
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from vbt.data.timeseries import load_source
def main():
 pred=np.load(ROOT/"outputs/phase3/sub-002/true_stim_source.npz")["source"]; target_path=Path("/home/hmzhang/remote/public_data/VEP_Cohort_v2.0/data/VirtualEpilepticCohort/derivatives/tvb/sub-002/ses-02/VEPhypothesis/sub-002_simulated_source_timeseries_run-01.npz"); _,target=load_source(target_path); n=min(len(pred),len(target)); a=pred[:n].reshape(-1); b=target[:n].reshape(-1); corr=float(np.corrcoef(a,b)[0,1]); report={"phase":"4","subject":"sub-002","target":str(target_path),"compared_samples":n,"descriptive_spatiotemporal_pearson":corr,"pearson_interpretation":"descriptive only; original RNG/TVB version unavailable","G4A_target_and_prediction_finite":"PASS" if np.isfinite(pred).all() else "FAIL","G4B_feature_comparison":"PARTIAL: source Pearson only","G4C_true_vs_control_null":"NOT TESTED: full permuted-location null not run"}; out=ROOT/"outputs/phase4/sub-002"; out.mkdir(parents=True,exist_ok=True); (out/"phase4_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
