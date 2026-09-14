# RAG prompt assembly example

*Produced by the `rag-pipeline` skill. The context block below is the top-5 output of hybrid retrieval + cross-encoder reranking over this project's own analysis reports.*

```text
You are answering questions about an internal data-analysis project.

Answer ONLY from the context below. If the answer is not in the context, say
"I don't know from the provided context." Cite every claim inline as [n].

CONTEXT
[1] source=metric_reconciliation.md section="Sources compared"
Sources compared. | # | Source | Definition | Total (GBP) | |---|---|---|---:| | A | Raw Kaggle feed (`online_retail.csv`) | `SUM(Quantity * UnitPrice)` over all 541,909 rows, no exclusions | 9,747,747.93 | | B | Analytics mart (`retail.db.invoice_lines`) | Same formula after the mart's documented exclusions | 8,911,407.90 | **Variance: GBP 836,340.03 (8.58% of source A).**

[2] source=query_business_logic.md section="Output columns"
Output columns. | Column | Business meaning | Edge cases | |---|---|---| | `CustomerID` | The customer's account number | Guest/unidentified checkouts are absent — they have no CustomerID and were excluded from the mart | | `Country` | Billing country | 'Unspecified' exists in the raw feed | | `orders` | Distinct invoices placed in 2011 | Cancelled orders (InvoiceNo starting 'C') are excluded from

[3] source=query_business_logic.md section="Questions for the query author"
Questions for the query author. 1. Should returns be netted off? The mart drops cancellations entirely, so `revenue` overstates net revenue by roughly the cancellation rate. 2. Should the ~25% of transactions with no CustomerID be represented as an 'unidentified' bucket rather than dropped? 3. Is calendar 2011 the intended window? The dataset starts 2010-12-01, so 2011 excludes the first month of 

[4] source=metric_reconciliation.md section="Verdict: RECONCILED"
Verdict: RECONCILED. Every pound of the variance is attributable to a documented, intentional mart exclusion. There is no data-loss or pipeline defect.

[5] source=metric_reconciliation.md section="Monthly detail"
Monthly detail. | Month | Source A raw | Source B mart | Diff | Diff % | |---|---:|---:|---:|---:| | 2010-12 | 748,957.02 | 572,713.89 | 176,243.13 | 23.53% | | 2011-01 | 560,000.26 | 569,445.04 | -9,444.78 | -1.69% | | 2011-02 | 498,062.65 | 447,137.35 | 50,925.30 | 10.22% | | 2011-03 | 683,267.08 | 595,500.76 | 87,766.32 | 12.85% | | 2011-04 | 493,207.12 | 469,200.36 | 24,006.76 | 4.87% | | 2011

QUESTION
What is the gross revenue variance between the raw feed and the mart?

ANSWER (with inline [n] citations):
```
