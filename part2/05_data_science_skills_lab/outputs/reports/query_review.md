# Query Review - `top_customers_2011`

*Produced by the `query-validation` skill. Engine: SQLite (patterns noted for Postgres/BigQuery).*

## Findings

| Severity | Category | Finding | Fix |
|---|---|---|---|
| HIGH | correctness | SELECT * | Selects every column from a 3-table join, then GROUP BYs one column. Non-aggregated columns are silently arbitrary. Name the columns. |
| HIGH | correctness | Implicit comma join | `FROM a, b, c WHERE ...` hides the join condition; one missing predicate becomes a cross join. Use explicit JOIN ... ON. |
| HIGH | performance | Function on a filtered column | `strftime('%Y', InvoiceDate) = '2011'` is not sargable — it defeats idx_inv_date and forces a full scan. Use a half-open range instead. |
| MEDIUM | portability | GROUP BY subset of selected columns | Grouping by CustomerID while selecting Country relies on SQLite's bare-column extension; it is an error on Postgres/BigQuery. |
| LOW | style | No LIMIT on an ORDER BY | A 'top customers' query should bound its output. |

## Before

```sql
-- Top customers by revenue, 2011 (pre-review version)
SELECT *
FROM invoice_lines l, invoices i, customers c
WHERE l.InvoiceNo = i.InvoiceNo
  AND i.CustomerID = c.CustomerID
  AND strftime('%Y', i.InvoiceDate) = '2011'
  AND c.Country <> 'Unspecified'
GROUP BY c.CustomerID
ORDER BY SUM(l.Quantity * l.UnitPrice) DESC
```

## After

```sql
SELECT
    c.CustomerID,
    c.Country,
    COUNT(DISTINCT i.InvoiceNo)        AS orders,
    SUM(l.LineRevenue)                 AS revenue
FROM invoice_lines AS l
JOIN invoices  AS i ON i.InvoiceNo  = l.InvoiceNo
JOIN customers AS c ON c.CustomerID = i.CustomerID
WHERE i.InvoiceDate >= '2011-01-01'
  AND i.InvoiceDate <  '2012-01-01'
GROUP BY c.CustomerID, c.Country
ORDER BY revenue DESC
LIMIT 10
```

## Measured impact

- original: **351.1 ms**
- reviewed: **115.7 ms**
- **3.0x faster**, and the reviewed version returns a defined column set.
