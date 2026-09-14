# QA sign-off

*Produced by the `analysis-qa-checklist` skill.*

**Decision: CONDITIONAL**

| Check | Status | Note |
|---|---|---|
| Question framing | PASS | Both tracks trace to a stated business question in Phase 1. |
| Data sourcing | PASS | Both datasets hashed (SHA-256) and version-pinned; the mart's exclusions are reconciled to the raw feed with zero residual. |
| Transformations | PASS | Every cleaning statistic is fit on train only; target encoding is cross-fitted. Leakage inflation measured at +0.081 corr. |
| Statistical validity | WARN | The tuned-vs-untuned CV gain (+0.016) is smaller than fold noise (+/-0.022). Reported as 'no measurable gain', not as an improvement. |
| Findings | WARN | Holdout PR-AUC 0.791 is materially below CV 0.867 on n=179. Stated explicitly rather than smoothed over. |
| Subgroup performance | FAIL | Male-passenger recall 0.417 is close to unusable. Blocks promotion to any automated decision. |
| Presentation | PASS | Every chart is annotated with its finding; the colourblind-safe Okabe-Ito palette is used throughout. |
| Assumptions documented | PASS | 9 cleaning decisions plus a separate assumptions log. |
| Reproducibility | PASS | Seeded, pinned, hashed; one command reruns the whole pipeline. |

The single FAIL (subgroup performance) blocks automated use. The analysis may be delivered as decision support with that caveat stated up front.
