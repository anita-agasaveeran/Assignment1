# Analysis brief - Online Retail

*Produced by the `stakeholder-requirements-gathering` skill.*

## 1. The request as received

> "Can you look into our customers? I want to know who the good ones are and why
> December looked so bad."

Two questions in one sentence, at different altitudes, with no success criteria.
This is exactly the vague ask the skill exists to convert into a scoped brief.

## 2. Intake interview - what was established

| Question | Answer |
|---|---|
| What decision does this inform? | Where to spend a fixed retention budget next quarter, and whether the December drop needs an incident response |
| Who is the audience? | Commercial VP (decides), analytics team (executes) |
| What does "done" look like? | A ranked, costed list of customer groups, and a yes/no on whether December was a real business event |
| Decision type | **Operational** (budget allocation) + **Tactical** (incident triage) - not strategic, so speed matters more than exhaustiveness |
| Deadline | End of week |
| What is explicitly out of scope? | Product-level assortment analysis; anything requiring marketing-spend data (we do not have it) |

## 3. Business questions, restated answerably

- **Q1.** Which customer groups exist, how large is each, and what share of revenue does each hold?
- **Q2.** How many customers return after a first order, and where in that sequence do we lose them?
- **Q3.** Is the December revenue fall a genuine business event or a data artefact?
- **Q4.** What is the single largest quantifiable opportunity, with a range?

## 4. Success criteria

1. Every group is defined by rules a marketer can act on, not by an opaque cluster id.
2. Every number traces to a query that can be re-run.
3. The December question gets a yes/no with evidence, not a hedge.
4. The opportunity is stated as a range with its assumptions named.

## 5. Scope

**In:** the 12-month transaction file; customer-level aggregation; revenue, orders, recency.
**Out:** CAC, LTV:CAC, payback - **the dataset contains no marketing spend**. This was
raised at intake rather than discovered at delivery, and no proxy will be invented.

## 6. Sign-off

Requirements confirmed before any data was touched. Any change to Q1-Q4 restarts this brief.
