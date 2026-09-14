# Dashboard specification - Customer Health

*Produced by the `dashboard-specification` skill.*

## 1. Purpose

> This dashboard answers **"is our customer base getting healthier or sicker?"** for the
> **Commercial VP and lifecycle marketing team**, who need to decide **where next month's
> retention spend goes**.

If it cannot be said in one sentence the scope is too wide. Note what this sentence
excludes: product performance, inventory, and acquisition-channel attribution all belong
on other dashboards.

## 2. Users

| Audience | Visits | Their question | Technical comfort |
|---|---|---|---|
| Commercial VP | monthly | "Is the base growing or shrinking in value?" | low - hero numbers only |
| Lifecycle marketing | weekly | "Which customers do I contact this week?" | medium - needs the drill-down |
| Analyst | ad hoc | "Why did that number move?" | high - needs the raw export |

These are different needs. The VP view and the marketing view are **separate tabs**, not
one page with more filters.

## 3. Metric hierarchy

**Hero row (4 numbers, nothing more)**
| Metric | Definition | Current | Target |
|---|---|---|---|
| Active customers (90d) | distinct CustomerID with an order in 90 days | 4,338 | growth MoM |
| Repeat rate | customers with >= 2 orders / all customers | 65.6% | >= 65% |
| AOV | gross revenue / distinct invoices | GBP 481 | stable |
| Champion revenue share | Champions revenue / total | 63% | watch - concentration risk |

**Supporting (second row)**
Revenue trend with a 14-day rolling mean; month-1 retention by cohort
(currently 21% mean); the repeat-purchase funnel;
segment mix over time.

**Detail (on drill-down only)**
Customer list with R/F/M and segment; per-country splits; the raw export.

Total: **8 distinct metrics**. The skill's ceiling is 10-12; more than that and the
dashboard is trying to be a warehouse.

## 4. Layout

```
+--------------------------------------------------------------+
|  HERO   Active customers | Repeat rate | AOV | Champion share |   <- top-left = most important
+--------------------------------------------------------------+
|  TREND  Revenue, daily + 14d rolling mean        [12 months]  |
+---------------------------------+----------------------------+
|  Cohort retention heatmap       |  Repeat-purchase funnel     |
+---------------------------------+----------------------------+
|  Segment mix over time (stacked area)                         |
+--------------------------------------------------------------+
|  > Customer detail table (collapsed by default)               |
+--------------------------------------------------------------+
```

## 5. Interactivity

| Control | Scope | Justification |
|---|---|---|
| Date range | global | Every metric is period-dependent |
| Country | global | The only dimension that meaningfully splits this business |
| Segment | global | It is the dashboard's organising idea |
| Click a segment -> customer list | drill-down | Turns a number into a work queue for marketing |

Rejected: product-category and channel filters. Neither serves the stated purpose, and
each one added is a maintenance cost and a source of contradictory numbers.

## 6. Data requirements

| Metric | Source | Transformation | Refresh |
|---|---|---|---|
| All revenue metrics | `invoice_lines` | `SUM(Quantity * UnitPrice)`, mart exclusions applied | daily 06:00 |
| Segments | `customers` + RFM job | k-means k=4, silhouette 0.381 | weekly |
| Cohorts | `invoices` | first-purchase month | daily |

**Mandatory guardrail:** the trend panel must **label the current period as incomplete**.
This dashboard's own analysis found a 70% "collapse" that was
purely a partial-month artefact. Any dashboard that can reproduce that mistake will.

## 7. Success criteria

- >= 60% of the lifecycle team open it weekly within 8 weeks.
- Ad-hoc "how many customers do we have" requests drop by half.
- Zero incidents caused by misreading a partial period (baseline: one, in this analysis).
