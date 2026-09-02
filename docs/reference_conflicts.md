# Reference boundary

Only `default` and `vep_25` are user profiles.

`vep_25` follows the executable seven-state `SpatEpiStim` path instantiated by
`Sim_SEEG_nf.ipynb` in Git tag `0.1.0` (`e8b6f597...`). The checked later
snapshot `5367ef4...` has byte-identical core model/integrator/notebook files.

The Nature Computational Science paper prints a different four-state `x,y,z,m`
model. It is retained only in `vbt.audit.paper4d` as a mathematical audit. The
repository's alternative four-state `Spat3DEpi` is not a pipeline: it uses
`50*Istim`, `25*abs(Istim)`, mutates `x0` in `dfun`, and declares `cvar=[0]`
while reading three coupling components.

The parcel VEP cohort has no original ~20k-vertex patient surface, geodesic
Laplace kernel, delay history, TVB monitor buffers, or original RNG. Therefore
`vep_25` means repo7D core-faithful parcel execution, not a full reproduction of
the high-resolution 2025 patient model.

The cohort's saved stimulation weights are normalized to a maximum of one and
are not the missing repository electric-field pattern. `vep_25` applies a
fixed `0.001` parcel input conversion for a finite demonstration run. This is
reported as `PARCEL_INPUT_APPROXIMATION`, and is excluded from exact trajectory
claims; exact RHS/Heun/trajectory gates use identical oracle inputs instead.
