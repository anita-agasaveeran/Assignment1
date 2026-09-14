# invoice_lines

## Overview

**Name:** `invoice_lines`  
**Type:** table  
**Domain:** [fill in]  
**Criticality:** [critical / high / medium / low]  

**Description:**  
[One sentence: what business process or entity does this table represent?]

## Ownership

- **Business Owner:** [name / team]
- **Technical Owner:** [name / team]

## Schema

**Row Count (at extraction):** 397,884  
**Extracted:** 2026-09-13T18:14:05.997344  

| Column | Type | Nullable | Keys | Description |
| --- | --- | --- | --- | --- |
| LineID | INTEGER | Yes | PK | [description] |
| InvoiceNo | TEXT | No | FK | [description] |
| StockCode | TEXT | No | FK | [description] |
| Quantity | INTEGER | No | - | [description] |
| UnitPrice | REAL | No | - | [description] |
| LineRevenue | REAL | No | - | [description] |
| InvoiceDate | TEXT | No | - | [description] |

## Data Quality

- **Completeness:** 100% on all columns (nulls were excluded upstream by the mart)
- **Freshness:** STATIC - this is a historical extract ending 2011-12-09, not a live feed
- **Duplicate rate:** 0.97% full-row duplicates exist in the *raw* source; the mart retains
  them because repeated identical lines on one invoice are legitimate in this data
- **Known issues:**
  - Excludes ~25% of raw transactions that have no CustomerID
  - Excludes cancellations (raw `InvoiceNo` prefixed `C`)
  - Revenue is GROSS. See `metric_reconciliation.md` for the 8.58% gap to the raw feed

## Business context

- **Business Owner:** Commercial (Analytics team as data steward)
- **Technical Owner:** Analytics Engineering
- **Criticality:** HIGH - every customer-level metric in the Customer Health dashboard
  resolves to this table
- **Business purpose:** one row per product line on one invoice. This is the finest grain
  at which revenue exists, and the base of all revenue aggregation.
- **Known use cases:** RFM segmentation, cohort retention, the repeat-purchase funnel,
  the daily revenue series

## Column notes

| Column | Business meaning | Valid values / gotchas |
|---|---|---|
| `LineID` | Surrogate key | Generated on load; carries no business meaning |
| `InvoiceNo` | The order this line belongs to | 6 digits. A `C` prefix means cancellation and is absent from this table |
| `StockCode` | Product identifier | Alphanumeric. Some codes (`POST`, `M`) are charges, not products |
| `Quantity` | Units on this line | Always positive here; the raw feed contains negatives |
| `UnitPrice` | Price per unit, GBP | Always > 0 here; the raw feed contains zeros |
| `LineRevenue` | `Quantity * UnitPrice` | Gross. Not net of returns, excludes shipping and tax |

## Lineage

**Upstream:** `data/raw/online_retail.csv` (Kaggle `carrie1/ecommerce-data`, = UCI 352),
via `scripts/p0_build_warehouse.py`.
**Downstream:** `rfm_segments`, `retention_matrix`, `funnel_repeat_purchase`,
the Customer Health dashboard, the LoRA fine-tuning dataset.

## Access & governance

- **Access level:** restricted (internal analytics)
- **Sensitivity:** pseudonymous - `CustomerID` is an account number, not a name, but it is
  re-identifying when joined to a CRM. Treat as **personal data** under GDPR.
- **Compliance tags:** GDPR (pseudonymous personal data)
- **Retention:** historical research extract; no live subject-access obligation
- **Access instructions:** read-only via the analytics warehouse; request through Analytics Eng
