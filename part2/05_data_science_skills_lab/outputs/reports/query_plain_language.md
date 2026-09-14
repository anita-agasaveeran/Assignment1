# SQL Explanation

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

### What is being calculated (SELECT)
- c.CustomerID
- c.Country
- Count distinct DISTINCT i.InvoiceNo (as orders)
- Sum of l.LineRevenue (as revenue)

### Data source (FROM)
- Start with: invoice_lines

### Filters applied (WHERE)
- i.InvoiceDate on or after 2011-01-01
- i.InvoiceDate less than 2012-01-01

### Grouping (GROUP BY)
- Calculate separately for each: c.CustomerID, c.Country

### Sorting (ORDER BY)
- Sort by: revenue DESC

### Validation questions
- Are these the correct filter conditions for the intended population?
- Does the GROUP BY grain match what a single row should represent?
- Are NULL values handled explicitly in aggregations?
