# VBT baseline: VEP synthetic brain digital twin

This repository is an auditable baseline for spontaneous seizure dynamics,
propagation, stimulation, reduced-Epileptor inversion, and EZN ranking on
`VEP_Cohort_v2.0`. It never modifies the VEP dataset or `VBT_INS_Stimulation`.

## Scientific profiles

| Profile | Purpose | Source | Coupling | Integrator |
| --- | --- | --- | --- | --- |
| `vep_cohort` | primary replay of VEP spontaneous data | `x2-x1` | Difference, raw-max SC | HeunStochastic, saved `dt/nsig`, colored noise |
| `legacy_spatepi` | diagnostic old VBT notebook path | monitored `u1-q1` | Heaviside, log-normalized SC | deterministic adapted Heun |

The legacy implementation is **not** the direct generator of the VEP
spontaneous cohort. The previous Phase-1 label `x1-x2` was incorrect for both
profiles: VEP spontaneous data store `x2-x1`; the custom VBT subtraction was
`u1-q1` (state indices 0 and 3), not indices 0 and 1.

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
