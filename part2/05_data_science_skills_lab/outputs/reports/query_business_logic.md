# What `top_customers_2011` actually calculates

*Produced by the `sql-to-business-logic` skill.*

**Business question:** which customers generated the most revenue in calendar 2011?

## Step by step

1. **Data combined** — every product line on every order (`invoice_lines`) is matched to its order header (`invoices`), and each order to the customer who placed it (`customers`). All three joins are inner joins, so a line only counts if its order and customer both exist in the mart. Orphan lines would be dropped silently — referential integrity was checked separately and is 0% orphaned.
2. **Rows included** — only orders dated on or after 1 Jan 2011 and before 1 Jan 2012. The boundary is half-open, so an order timestamped 31 Dec 2011 23:59 is included and 1 Jan 2012 00:00 is not.
3. **Grouping** — one output row per customer (the customer's country travels along because a customer has exactly one country in this mart).
4. **Measures** — `orders` counts *distinct* invoices, so a customer who bought 40 products on one order counts as one order, not 40. `revenue` sums quantity x unit price across every line.
5. **Output** — the 10 customers with the highest 2011 revenue, highest first.

## Output columns

| Column | Business meaning | Edge cases |
|---|---|---|
| `CustomerID` | The customer's account number | Guest/unidentified checkouts are absent — they have no CustomerID and were excluded from the mart |
| `Country` | Billing country | 'Unspecified' exists in the raw feed |
| `orders` | Distinct invoices placed in 2011 | Cancelled orders (InvoiceNo starting 'C') are excluded from the mart, so this is gross orders, not net |
| `revenue` | Gross revenue, GBP | Gross, not net of returns. Excludes shipping/tax. |

## Questions for the query author

1. Should returns be netted off? The mart drops cancellations entirely, so `revenue` overstates net revenue by roughly the cancellation rate.
2. Should the ~25% of transactions with no CustomerID be represented as an 'unidentified' bucket rather than dropped?
3. Is calendar 2011 the intended window? The dataset starts 2010-12-01, so 2011 excludes the first month of data.
4. Is `Country` the billing or shipping country?
5. Should this be gross revenue or gross margin?
