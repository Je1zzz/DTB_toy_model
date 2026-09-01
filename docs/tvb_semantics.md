# TVB compatibility notes

The VEP generator invokes TVB's six-state `Epileptor`, `Difference` coupling,
`HeunStochastic`, additive colored noise with `ntau=1`, and
`TemporalAverage(period=1)`. Saved simulator TSVs have `dt=.05`; monitor period
is not the integration step.

Spontaneous coupling variables are state indices 0 and 3 (`x1`, `x2`), and the
stored source is `x2-x1`. Connectivity is zero-diagonal and divided by its raw
maximum; the cohort generator does not use `log1p`.

The local colored-noise compatibility layer is provisional because the exact
generation-era TVB version and original RNG state are not recorded. Equation
and deterministic integration gates can be checked; point-wise stochastic
replay cannot be claimed from the available provenance.

Primary generator files are `virtual_epileptic_seeg_ret_patient.py`,
`STIM_virtual_epileptic_seeg_ret.py`, and
`utils_functions/model_2populations.py`, at commit
`a99c88354015f4c961d49a92c076ed0b675c740b`.

The stimulation generator has an explicit upstream inconsistency:
`EpileptorStim2Populations.cvar=[0]`, while `_numba_dfun` reads `c_pop[0]`,
`c_pop[1]`, and `c_pop[2]`. The baseline uses the sole x1 Difference component
for those terms and marks stimulation equation equivalence NOT FULLY VERIFIED.
