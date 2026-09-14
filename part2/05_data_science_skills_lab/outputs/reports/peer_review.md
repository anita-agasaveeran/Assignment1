# Peer review

*Produced by the `peer-review-template` skill.*

**Scope agreed with author:** statistical validity, leakage, and honesty of reporting.

### [MUST-FIX] Male-passenger recall is 0.417. Do not promote to any automated decision until either a group-specific threshold is set or the gap is closed.

**Author response:** Accepted - recorded as the blocking item in the QA sign-off.

### [MUST-FIX] The tuned-vs-untuned CV delta (+0.016) is inside fold noise (+/-0.022) but the tuned model was still promoted.

**Author response:** Accepted - the report now states there is no measurable gain and that the tuned model was kept for reproducibility of the search, not for accuracy.

### [SHOULD-FIX] Titanic families share tickets; a random split can split a family. GroupKFold on ticket would be more conservative.

**Author response:** Acknowledged - logged as open assumption #5. Not changed in this pass; it would alter every number in the report.

### [SHOULD-FIX] The 45% gross margin behind the LTV figure is an industry placeholder.

**Author response:** Acknowledged - logged as open assumption #3 and flagged inline wherever LTV appears.

### [SHOULD-FIX] The A/B test section uses a simulated assignment.

**Author response:** Accepted - the simulation is labelled in the console output, the JSON (`simulated: true`) and the report.

### [OPTIONAL] The cross-encoder reranker reduced retrieval MRR and was still built.

**Author response:** Accepted - kept deliberately, because the negative result is the finding.

### [OPTIONAL] ARIMA(7,1,1) order was chosen by inspection rather than by AIC search.

**Author response:** Acknowledged - MAPE 28% is reported so the reader can judge; a grid over (p,d,q) is the obvious next step.

## Sign-off

2 must-fix items raised, both addressed. Deliverable as decision support; **not** approved for automated decisioning while the subgroup gap stands.
