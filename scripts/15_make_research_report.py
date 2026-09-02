#!/usr/bin/env python
"""Combine executed evidence into the personalization/stimulation research report."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
load=lambda path: json.loads((ROOT/path).read_text())
p=load(Path("outputs/personalization_map/summary.json")); existing=load(Path("outputs/stimulation_existing/summary.json")); geometry=load(Path("outputs/stimulation_existing/location_policy_geometry.json")); cf=load(Path("outputs/stimulation_counterfactual/summary.json")); stats=load(Path("outputs/stimulation_counterfactual/statistics.json"))
nuts=load(Path("outputs/timing_nuts/sub-001/blind_run_report.json"))
replay=load(Path("outputs/replay_ev/summary.json")); heldout=load(Path("outputs/heldout_seizure_ev/summary.json")); oracle=load(Path("outputs/oracle_personalization/summary.json"))
report={
 "data_scope":{"subjects":30,"synthetic_seeg_not_clinical":True,"seizure_subjects":30,"seizure_recordings":53,"interictal_subjects":30,"stimulation_subjects":14,"stimulation_recordings":16},
 "result_1_inferred_personalization":{"finding":"short MAP personalization did not outperform population controls","metrics":p["metrics"],"paired_statistics":p["paired_statistics"],"timing_seconds":p["timing_seconds"],"interpretation":"direct motivation to improve personalization/inversion; not evidence that personalization is unnecessary"},
 "inversion_timing_example":{"map_30_subject_summary":p["timing_seconds"],"nuts_sub001_smoke":nuts["timing_seconds"],"nuts_engineering_status":nuts["engineering_status"],"nuts_inference_quality_status":nuts["inference_quality_status"],"nuts_rhat_x0_max":nuts["rhat_x0_max"]},
 "result_2_existing_stimulation":{"finding":"provided synthetic stimulation responses were weakly EZ-selective on average","metrics":existing["metrics"],"timing_seconds":existing["timing_seconds"]},
 "result_3_location_geometry":{"finding":"candidate site strongly changes anatomical EZ engagement","evidence":geometry},
 "result_4_counterfactual_location":{"finding":"at matched synthetic dose, oracle EZ-aware location improved EZ-selective response relative to random locations","policy_response_auroc_mean":cf["policy_response_auroc_mean"],"statistics":stats["comparisons"],"timing_seconds":cf["timing_seconds"],"scope":cf["scope"]},
 "result_5_oracle_personalization_ablation":oracle,
 "result_6_replay_ev":{"in_sample":replay,"heldout_synthetic_seizures":heldout},
 "result_7_inferred_evn":{"rule":cf["inferred_evn_rule"],"gate":cf["inferred_policy_gate"],"response_auroc_mean":cf["policy_response_auroc_mean"]["inferred"],"paired_vs_random":[x for x in stats["comparisons"] if x["comparison"].startswith("inferred")][0]},
 "why_personalization":{"question":"Can subject-specific parameters beat population/default/mismatched twins and predict held-out recordings?","current_answer":"current short MAP has good in-sample feature EV but weak held-out EV and poor EZN localization; robust personalization and converged inversion remain necessary"},
 "why_next_stimulation":{"question":"Does selecting location change EZN-selective response under matched dose?","current_answer":"yes for the synthetic oracle policy, but current inferred-EVN is worse than random on average and fails the inversion-quality gate"},
 "remaining_primary_experiments":["converged multi-chain Bayesian inversion","posterior predictive calibration on frozen held-out seizures","retest inferred-EVN only after inversion and EV gates pass"],
 "forbidden_claims":["real clinical SEEG accuracy","therapeutic seizure suppression","clinical safety","clinically validated next stimulation"]
}
out=ROOT/"outputs/DTB_PERSONALIZATION_STIMULATION_REPORT.json"; out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
