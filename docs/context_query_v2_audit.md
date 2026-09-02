# Context-query v2 audit

## Frozen decision

- `A3873C8_ENGINEERING = AUDITABLE_PARTIAL_PASS`
- `END_TO_END_DELTA_MAP = IMPLEMENTED_NOT_RUN_ON_VEP`
- `SPARSE_EZN_REPRESENTATION_V1 = FAIL_BY_PREDECLARED_SPEARMAN`
- `SPARSE_EZN_REPRESENTATION_V1_1 = PASS_REPRESENTATION_ONLY`
- `DELTA_SEEG_PERSONALIZATION_INFORMATION_V1 = FROZEN_FAIL`
- `PROTOCOL_V2_FISHER_OED = EXECUTED_FAIL_PRE_ORACLE`
- `TRAJECTORY_PERSONALIZATION_BASELINE = FAIL_AT_INFORMATION_GATE`
- `EZN_SCIENTIFIC_CLAIM = STOP`
- `S3_UNSEEN_CONDITION = INELIGIBLE`

## Engineering corrections

The context optimizer now compares candidate-specific stimulated-minus-control
SEEG against observed delta SEEG. Coupling is fixed at 0.5 by default and is
removed from the benchmark parameter vector. Data and posterior Jacobian
singular values are reported separately, with data effective rank defined by
`s_i / s_max >= 1e-6`. Gain row names must match observation channel names in
exact order. The spectral coefficient box conservatively guarantees expanded
ROI x0 remains in `[-3.5, -1.0]`; diagnostics report ROI-level boundary hits.
An anti-leakage test confirms that query targets cannot affect context fitting.

## Tie-aware EZN representation ceiling v1.1

The v1 configuration was unchanged: rank 20, sparse scale 0.15, epsilon 0.001,
and representation noise 0.05. Only the invalid tie-sensitive gate was replaced.

| Metric | Result | Gate |
| --- | ---: | ---: |
| Median x0 RMSE | 0.01399 | <= 0.10 |
| Median meaningful-pair concordance | 1.000 | >= 0.95 |
| Subjects with concordance >= 0.90 | 30/30 | >= 27/30 |
| Tie-grouped macro-AUPRC | 0.9892 | >= 0.90 |
| Fractional Recall at true EZ count | 0.9695 | >= 0.90 |
| Median Spearman, report only | 0.3523 | not a gate |

This is a parameter representation ceiling without SEEG inversion. It cannot
support a scientific EZN-identification claim.

## Fisher/OED v2

Site design used only patient SC, patient Gain, LOSO population x0 and the
frozen two-state model. It used rank-20 prior-standardized graph coefficients,
fixed coupling 0.5, central differences at 0.01 with a 0.005 check, and samples
from steps 200 through 1999. All candidate Fisher hashes and chosen folds are
stored in `outputs/context_query/fisher_oed_v2/subject_designs.json`.

The median improvement in worst-fold log determinant over v1 was 8.5487, above
the required `log(10)`. Chosen-site derivative instability was below
`3.1e-6`, and posterior condition numbers were below 1,315. Nevertheless every
one of the 18 two-context folds had effective dimension only 1.987 to 1.996,
far below the required 10. Thus the pre-oracle gate failed.

Per the preregistered stopping rule, oracle response information was not read,
the six-subject MAP benchmark was not run, and changing sites, pulse amplitude,
duration, or adding protocol v3 is prohibited. This is a credible negative
result about information content under the frozen reduced trajectory model,
not evidence that personalization or EZN recovery works clinically.
