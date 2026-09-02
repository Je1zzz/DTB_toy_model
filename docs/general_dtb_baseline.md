# General DTB baseline

The pipeline is `data -> anatomy/connectome -> forward -> SEEG -> features ->
inversion -> posterior/source -> EV/EZN -> validation`.

Forward dynamics and inversion are independent choices. `default` and
`vep_25` select the forward implementation; `map` and `nuts` select an
implemented inversion method. ADVI, gradient MAP, EnKF and SBI are visible in
`python -m vbt.cli methods`, but fail with `NOT_IMPLEMENTED` until they have a
tested implementation.

## What the VEP cohort does not contain

| Class | Examples | Patient-specific? | Can code recreate it without raw inputs? |
| --- | --- | --- | --- |
| raw measurements | T1/T2, DWI, implant CT/MRI, clinical SEEG, stimulation log | yes | no |
| derived anatomy/operators | cortical surface, region map, tractography SC and delays, electrode gain, electric stimulation field | yes, derived from the raw measurements plus modeling assumptions | no |
| general methods | Laplace kernel, neural-mass equations, integrator, monitor | no | yes, but matching old behavior needs versions and parameters |
| historical runtime state | RNG state, vertex initial state, delay buffer, exact software stack | run-specific | normally no |

The Laplace formula is general code; its sparse matrix is patient-specific
because it depends on that patient's reconstructed mesh and geodesic distances.
Likewise, conduction speed is a model parameter, whereas the tract lengths and
therefore delays are patient-derived.

## Readiness levels

1. `cohort_synthetic`: synthetic truth, SEEG, SC and gain.
2. `normative_connectome_patient_seeg`: real SEEG/electrodes with a declared
   normative SC; never call this a patient-specific connectome.
3. `patient_specific_parcel`: patient T1/DWI, parcellation, SC, gain and SEEG.
4. `patient_specific_surface`: additionally requires surface, region map, local
   connectivity and stimulation field.

Passing an engineering run never upgrades readiness or establishes clinical
EZN validity.
