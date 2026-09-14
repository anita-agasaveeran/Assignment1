# Metric Reconciliation - Gross Revenue

*Produced by the `metric-reconciliation` skill.*

## Sources compared

| # | Source | Definition | Total (GBP) |
|---|---|---|---:|
| A | Raw Kaggle feed (`online_retail.csv`) | `SUM(Quantity * UnitPrice)` over all 541,909 rows, no exclusions | 9,747,747.93 |
| B | Analytics mart (`retail.db.invoice_lines`) | Same formula after the mart's documented exclusions | 8,911,407.90 |

**Variance: GBP 836,340.03 (8.58% of source A).**

## Discrepancy decomposition

| Cause | Amount (GBP) | Share of gap |
|---|---:|---:|
| Cancellations (InvoiceNo prefix 'C') | -896,812.49 | -107.2% |
| No CustomerID (guest/unidentified) | 1,733,152.52 | 207.2% |
| Zero/negative price or quantity (non-cancel, identified) | 0.00 | 0.0% |
| **Unexplained residual** | **-0.00** | **-0.0%** |

## Verdict: RECONCILED

Every pound of the variance is attributable to a documented, intentional mart exclusion. There is no data-loss or pipeline defect.

## Which number should be reported?

- **Customer-level analysis** (cohorts, RFM, LTV) must use **source B** — source A's unidentified rows cannot be attributed to a customer at all.
- **Company-level gross revenue** should use a third definition not yet built: source A *net of* cancellations, i.e. keeping unidentified sales but subtracting returns. Reporting source B as 'total revenue' understates the business by 8.6%.
- **Recommended action:** add a `net_revenue` metric to the semantic layer so the distinction is codified rather than rediscovered each quarter.

## Monthly detail

| Month | Source A raw | Source B mart | Diff | Diff % |
|---|---:|---:|---:|---:|
| 2010-12 | 748,957.02 | 572,713.89 | 176,243.13 | 23.53% |
| 2011-01 | 560,000.26 | 569,445.04 | -9,444.78 | -1.69% |
| 2011-02 | 498,062.65 | 447,137.35 | 50,925.30 | 10.22% |
| 2011-03 | 683,267.08 | 595,500.76 | 87,766.32 | 12.85% |
| 2011-04 | 493,207.12 | 469,200.36 | 24,006.76 | 4.87% |
| 2011-05 | 723,333.51 | 678,594.56 | 44,738.95 | 6.19% |
| 2011-06 | 691,123.12 | 661,213.69 | 29,909.43 | 4.33% |
| 2011-07 | 681,300.11 | 600,091.01 | 81,209.10 | 11.92% |
| 2011-08 | 682,680.51 | 645,343.90 | 37,336.61 | 5.47% |
| 2011-09 | 1,019,687.62 | 952,838.38 | 66,849.24 | 6.56% |
| 2011-10 | 1,070,704.67 | 1,039,318.79 | 31,385.88 | 2.93% |
| 2011-11 | 1,461,756.25 | 1,161,817.38 | 299,938.87 | 20.52% |
| 2011-12 | 433,668.01 | 518,192.79 | -84,524.78 | -19.49% |