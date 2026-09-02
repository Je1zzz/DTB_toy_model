#!/usr/bin/env python
"""Consolidate single-patient traditional-personalization latency evidence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/personalization_latency"


def load(path):
    return json.loads((ROOT/path).read_text())


preflights=[]
for path in sorted(OUT.glob("prepare_run_*/preflight.json")):
    item=json.loads(path.read_text())
    preflights.append(item["stage_timing_seconds"])
if len(preflights)<2:
    raise RuntimeError("At least two timed preparation runs are required")

keys=list(preflights[0])
rows=[]
for key in keys:
    values=np.asarray([item[key] for item in preflights])
    rows.append({"stage":key,"runs":len(values),"median_seconds":float(np.median(values)),"min_seconds":float(values.min()),"max_seconds":float(values.max())})

map_summary=load(Path("outputs/personalization_map/summary.json"))
nuts=load(Path("outputs/timing_nuts/sub-001/blind_run_report.json"))
geometry=load(Path("outputs/stimulation_existing/location_policy_geometry.json"))
stimulation=load(Path("outputs/stimulation_counterfactual/summary.json"))

map_timing=map_summary["timing_seconds"]
map_post=map_timing["per_subject_total_mean"]-map_timing["data_features_mean"]-map_timing["optimization_mean"]
candidate_per_subject=geometry["timing_seconds"]["total"]/geometry["subjects"]
actions_per_subject=stimulation["actions"]/stimulation["subjects"]
forward_per_action=stimulation["timing_seconds"]["per_subject_mean"]/actions_per_subject

rows.extend([
    {"stage":"map_data_and_features_cohort_mean","runs":30,"median_seconds":map_timing["data_features_mean"],"min_seconds":None,"max_seconds":None},
    {"stage":"map_optimization_2x50_cohort_mean","runs":30,"median_seconds":map_timing["optimization_mean"],"min_seconds":None,"max_seconds":None},
    {"stage":"map_ezn_export_residual_cohort_mean","runs":30,"median_seconds":map_post,"min_seconds":None,"max_seconds":None},
    {"stage":"nuts_smoke_optimization_2x50","runs":1,"median_seconds":nuts["timing_seconds"]["optimization"],"min_seconds":None,"max_seconds":None},
    {"stage":"nuts_smoke_sampling_2chains_5warmup_5draws","runs":1,"median_seconds":nuts["timing_seconds"]["sampling"],"min_seconds":None,"max_seconds":None},
    {"stage":"nuts_smoke_ezn_postprocess","runs":1,"median_seconds":nuts["timing_seconds"]["postprocess"],"min_seconds":None,"max_seconds":None},
    {"stage":"stimulation_candidate_geometry_per_subject","runs":geometry["subjects"],"median_seconds":candidate_per_subject,"min_seconds":None,"max_seconds":None},
    {"stage":"stimulation_forward_per_action_amortized","runs":stimulation["actions"],"median_seconds":forward_per_action,"min_seconds":None,"max_seconds":None},
    {"stage":"stimulation_forward_23_actions_per_subject","runs":stimulation["subjects"],"median_seconds":stimulation["timing_seconds"]["per_subject_mean"],"min_seconds":None,"max_seconds":None},
])

frame=pd.DataFrame(rows); OUT.mkdir(parents=True,exist_ok=True); frame.to_csv(OUT/"traditional_stage_latency.csv",index=False)
report={
    "question":"How long does one traditional patient-personalization and next-stimulation workflow take?",
    "units":"wall-clock seconds",
    "hardware_context":"remote cuhk139 runtime; timings are implementation-specific",
    "prepare_repeats":len(preflights),
    "traditional_map":{"configuration":"2 L-BFGS starts x 50 iterations; point estimate only","single_patient_total_mean_seconds":map_timing["per_subject_total_mean"],"optimization_seconds":map_timing["optimization_mean"],"optimization_fraction":map_timing["optimization_mean"]/map_timing["per_subject_total_mean"]},
    "traditional_bayesian_smoke":{"configuration":"2 L-BFGS starts x 50; 2 NUTS chains; 5 warmup + 5 draws per chain","single_patient_total_seconds":nuts["timing_seconds"]["total"],"sampling_seconds":nuts["timing_seconds"]["sampling"],"sampling_fraction":nuts["timing_seconds"]["sampling"]/nuts["timing_seconds"]["total"],"quality_status":nuts["inference_quality_status"],"interpretation":"lower-bound runtime demonstration, not a converged traditional Bayesian run"},
    "next_stimulation":{"candidate_geometry_per_patient_seconds":candidate_per_subject,"evaluated_actions_per_patient":actions_per_subject,"forward_per_action_amortized_seconds":forward_per_action,"forward_all_actions_per_patient_seconds":stimulation["timing_seconds"]["per_subject_mean"]},
    "end_to_end":{"map_to_ezn_seconds":map_timing["per_subject_total_mean"],"map_plus_geometry_only_selection_seconds":map_timing["per_subject_total_mean"]+candidate_per_subject,"map_plus_23_action_forward_optimization_seconds":map_timing["per_subject_total_mean"]+candidate_per_subject+stimulation["timing_seconds"]["per_subject_mean"],"nuts_smoke_plus_23_action_forward_optimization_seconds":nuts["timing_seconds"]["total"]+candidate_per_subject+stimulation["timing_seconds"]["per_subject_mean"]},
    "fast_personalization_targets":["replace iterative MAP/NUTS inference or amortize it with a trained encoder","retain explicit EZN uncertainty/calibration gate","batch stimulation candidates because shared burn-in makes per-action cost sublinear","do not count optional QC rendering as algorithm latency"],
    "scientific_scope":"synthetic VEP benchmark; not clinical deployment latency",
}
(OUT/"traditional_latency_report.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
