# VBT baseline: VEP synthetic brain digital twin

This repository is an auditable baseline for spontaneous seizure dynamics,
propagation, stimulation, reduced-Epileptor inversion, and EZN ranking on
`VEP_Cohort_v2.0`. It never modifies the VEP dataset or `VBT_INS_Stimulation`.

## Scientific profiles

| Profile | Spontaneous | Stimulation | Purpose |
| --- | --- | --- | --- |
| `default` | cohort 6D, `x2-x1` | cohort 7D, Difference | smallest stable parcel baseline |
| `vep_25` | same cohort 6D engine | repo `SpatEpiStim` 7D, `u1-q1`, Heaviside | repository-core fidelity at parcel level |

These are the only selectable profiles. `vep_25` freezes the canonical notebook
settings (`dt=.2`, monitor period 10, `tt=.17`, `tau0=1000`, `tau3=600`) and
Git tag `0.1.0` (`e8b6f597...`). It exactly targets the tested repo7D RHS and
Heun semantics, not the different four-state equation printed in the paper.
See `docs/reference_conflicts.md`.

Stimulated VEP recordings use a separate seven-state compatibility model,
`EpileptorStim2Populations`, with accumulation state `m`. Saved run parameters
have priority over notebook examples.

## Reference priority

1. The parameter TSV/NPZ belonging to the selected VEP recording.
2. VEP generator commit `a99c88354015f4c961d49a92c076ed0b675c740b`.
3. VBT reference commit `5367ef4afb976271ecc6297b70a95dd3d944af61`
   for stimulation theory and the exact reduced two-state Stan model.
4. TVB upstream for equation and integrator semantics.

Forward seed 0 is a baseline reproducibility seed, not the unavailable original
cohort RNG seed. Point-wise stochastic waveform reproduction is not claimed.

## Phases

- 1B: corrected six-state replay and long-window diagnostics.
- 2: provided-source and replay propagation.
- 3: saved biphasic stimulation and true/zero controls.
- 4: stimulated-recording forward comparison.
- 5: exact fixed-tau reduced-Epileptor Stan inversion.
- 6: posterior mean `x0` to EZN score; larger `x0` ranks higher.
- 7: 30-subject mechanically blind benchmark, with separate prediction and
  truth-unlock processes and pre-unlock prediction hashes.

Blind preprocessing reads only BrainVision observations, SC, contact geometry,
and gain. `eig` is the right-singular-vector basis computed solely from gain;
it is not derived from x0, EZ/PZ, source truth, or a hypothesis heatmap. The
blind engine returns before any epileptor-parameter truth file is opened.

Phase 7 freezes 50 optimization starts, best 8, two chains per initialization,
500 warmup, 500 samples, `adapt_delta=.99`, `max_depth=7`, and seed 12345.
CPU concurrency is capped at four to protect the shared server. A later PASS
never erases an earlier FAIL; legacy adaptive-tau results are comparator-only.

## Run

```bash
cd /home/hmzhang/remote/项目/脑数字孪生/methods/VBT_baseline
PY=/data_hdd/hmzhang/env/tongji/bin/python
$PY -m unittest discover -s tests -v
$PY scripts/run_profile.py --profile default --subject sub-002
$PY scripts/run_profile.py --profile vep_25 --subject sub-002
$PY scripts/09_verify_repo_equivalence.py
$PY scripts/01b_reference_spontaneous.py
$PY scripts/01c_diagnose_long_window.py
$PY scripts/02_propagation.py
$PY scripts/03_stimulation_smoke.py
$PY scripts/04_validate_stimulated.py
$PY scripts/05a_prepare_inference.py
$PY scripts/05b_run_reference_inference.py
$PY scripts/05c_check_inference.py
$PY scripts/06_evaluate_ezn_dev.py
$PY scripts/07a_freeze_benchmark.py
$PY scripts/07b_run_blind_inference.py --jobs 1
$PY scripts/07c_unlock_truth_and_evaluate.py
$PY scripts/07d_make_final_report.py
```

Engineering completion is not clinical validation. Synthetic results do not
establish clinical EZN identification, treatment efficacy, or generalization.

## General DTB interface

Forward and inversion are independent selections:

```bash
PYTHONPATH=src $PY -m vbt.cli methods
PYTHONPATH=src $PY -m vbt.cli audit-data
PYTHONPATH=src $PY -m vbt.cli forward --profile default --subject sub-002
PYTHONPATH=src $PY -m vbt.cli infer --profile default --inversion map --subject sub-001
PYTHONPATH=src $PY -m vbt.cli infer --profile vep_25 --inversion nuts --subject sub-001
```

The two implemented inversion methods share one reduced-Epileptor Stan model:
`map` is an L-BFGS point-estimate baseline; `nuts` produces posterior samples
and must pass R-hat/ESS/BFMI/divergence/treedepth checks before scientific use.
See `docs/general_dtb_baseline.md` for data-readiness gates.
