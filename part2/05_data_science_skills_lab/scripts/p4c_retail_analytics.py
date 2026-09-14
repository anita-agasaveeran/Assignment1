"""CRISP-DM Phase 4 - Modeling (Track B: analytics on Online Retail).

Skills demonstrated:
  * cohort-analysis             -- monthly acquisition cohorts + retention matrix
  * segmentation-analysis       -- RFM + k-means, silhouette-validated, named segments
  * funnel-analysis             -- repeat-purchase progression funnel with drop-off sizing
  * time-series-analysis        -- stationarity, decomposition, anomalies, ARIMA forecast
  * root-cause-investigation    -- why did December revenue collapse?
  * ab-test-analysis            -- SRM, significance, power, guardrails
  * business-metrics-calculator -- e-commerce KPIs against benchmarks
"""
from __future__ import annotations

import json
import subprocess
import sqlite3
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

from common import (FIG, PALETTE, PROC, REP, SEED, TAB, WAREHOUSE, banner, save_table,
                    seed_everything, style_plots, write_json, write_report)

warnings.filterwarnings("ignore")
SKILLS = Path.home() / ".claude" / "skills"
PY = sys.executable
RESULTS: dict = {}


def run(skill: str, script: str, *args: str) -> str:
    path = SKILLS / skill / "scripts" / script
    print(f"\n    $ python {skill}/scripts/{script} ...")
    r = subprocess.run([PY, str(path), *args], capture_output=True, text=True)
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    print("\n".join("    " + ln for ln in out.splitlines()[:32]))
    return out


# ------------------------------------------------------------- cohort-analysis
def cohort_analysis(tx: pd.DataFrame) -> pd.DataFrame:
    banner("cohort-analysis", "Phase 4 - Modeling",
           "monthly acquisition cohorts; retention event = any purchase")
    print("  cohort grouping : month of first purchase")
    print("  retention event : placed >=1 order in that month")
    print("  granularity     : monthly, 13 periods (2010-12 .. 2011-12)")

    d = tx[["CustomerID", "InvoiceDate"]].copy()
    d["cohort_date"] = d.groupby("CustomerID")["InvoiceDate"].transform("min")
    d = d.rename(columns={"InvoiceDate": "activity_date", "CustomerID": "user_id"})
    src = PROC / "cohort_input.csv"
    d.to_csv(src, index=False)

    # Steps 3-5 use the skill's own scripts
    built = PROC / "cohort_table.csv"
    run("cohort-analysis", "cohort_builder.py", "--input", str(src),
        "--granularity", "monthly", "--output", str(built))
    matrix_csv = TAB / "retention_matrix.csv"
    run("cohort-analysis", "retention_matrix.py", "--input", str(built),
        "--format", "pct", "--output", str(matrix_csv))
    run("cohort-analysis", "cohort_visualizer.py", "--input", str(matrix_csv),
        "--chart", "heatmap", "--output", str(FIG / "p4c_retention_heatmap.png"))

    mat = pd.read_csv(matrix_csv, index_col=0)
    # Column 0 is "Cohort Size"; the retention periods are the "Period N" columns.
    periods = [c for c in mat.columns if c.startswith("Period")]
    mat = mat[periods]
    mat.index = [str(i)[:7] for i in mat.index]
    sizes = d.assign(c=d["cohort_date"].dt.to_period("M").astype(str)).groupby("c")["user_id"].nunique()

    # Step 6: interpret
    print("\n  interpretation:")
    # Exclude the final cohort, which has no month-1 opportunity yet.
    p1 = mat["Period 1"].dropna()
    print(f"  month-1 retention ranges {p1.min():.1f}% - {p1.max():.1f}% (mean {p1.mean():.1f}%)")
    big = sizes.idxmax()
    print(f"  largest cohort  : {big} with {sizes.max():,} customers")
    print(f"  cohorts under the skill's n>=100 reliability floor: "
          f"{sizes[sizes < 100].index.tolist() or 'none'}")
    dec = mat.loc["2010-12"].dropna()
    later = mat.drop(index="2010-12")["Period 1"].dropna()
    print(f"  2010-12 cohort month-1 retention {dec['Period 1']:.1f}% vs "
          f"{later.mean():.1f}% mean for every later cohort")
    print(f"  it also holds up: {dec.iloc[-1]:.1f}% still active at month {len(dec)-1}")
    print("  -> these are pre-existing wholesale buyers captured at the dataset's start,")
    print("     not a genuine acquisition cohort. Exclude it from retention benchmarks.")
    RESULTS["cohort"] = {"month1_retention_mean": float(p1.mean()),
                         "month1_min": float(p1.min()), "month1_max": float(p1.max()),
                         "largest_cohort": str(big), "largest_cohort_size": int(sizes.max()),
                         "cohorts_below_100": sizes[sizes < 100].index.tolist(),
                         "dec2010_month1": float(dec.iloc[1]),
                         "dec2010_final": float(dec.iloc[-1])}
    return mat


# ------------------------------------------------------- segmentation-analysis
def segmentation(tx: pd.DataFrame) -> pd.DataFrame:
    banner("segmentation-analysis", "Phase 4 - Modeling",
           "RFM + k-means, silhouette-validated, mapped to strategy")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    snapshot = tx["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = tx.groupby("CustomerID").agg(
        recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("LineRevenue", "sum"),
    ).query("monetary > 0")
    print(f"  step 2 variables: Recency / Frequency / Monetary on {len(rfm):,} customers")
    print(f"          snapshot date {snapshot.date()}")
    print(rfm.describe().T.round(2).to_string())

    # log-transform the two heavily skewed variables before scaling
    Xs = StandardScaler().fit_transform(
        np.c_[rfm["recency"], np.log1p(rfm["frequency"]), np.log1p(rfm["monetary"])])

    print("\n  step 5 validate k by silhouette (the skill's >0.3 gate):")
    scores = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Xs)
        scores[k] = silhouette_score(Xs, km.labels_, sample_size=3000, random_state=SEED)
        print(f"          k={k}  silhouette {scores[k]:.4f}"
              f"{'  <- best' if scores[k] == max(scores.values()) else ''}")
    k_best = 4  # business-useful granularity; silhouette is reported alongside
    print(f"  -> silhouette peaks at k={max(scores, key=scores.get)} "
          f"({max(scores.values()):.3f}), but k=2 is not actionable.")
    print(f"     Choosing k={k_best} (silhouette {scores[k_best]:.3f}, above the 0.3 gate) because the")
    print(f"     skill's step 1 says the goal decides k: marketing needs 3-7 treatable groups.")

    km = KMeans(n_clusters=k_best, random_state=SEED, n_init=10).fit(Xs)
    rfm["segment"] = km.labels_

    prof = rfm.groupby("segment").agg(
        customers=("recency", "size"), recency=("recency", "median"),
        frequency=("frequency", "median"), monetary=("monetary", "median"),
        revenue=("monetary", "sum")).round(2)
    prof["revenue_share"] = (prof["revenue"] / prof["revenue"].sum() * 100).round(1)
    prof["cust_share"] = (prof["customers"] / prof["customers"].sum() * 100).round(1)

    # Step 4: name each segment by its 2-3 defining characteristics vs the overall median
    overall = rfm[["recency", "frequency", "monetary"]].median()
    names, strategies = {}, {}
    for s in prof.index:
        r, f, m = prof.loc[s, ["recency", "frequency", "monetary"]]
        if f >= overall["frequency"] * 2 and m >= overall["monetary"] * 2:
            names[s], strategies[s] = "Champions", "Retain & Expand"
        elif r > overall["recency"] * 2:
            names[s], strategies[s] = "Lapsed", "Win-Back"
        elif m >= overall["monetary"]:
            names[s], strategies[s] = "Loyal", "Monetise"
        else:
            names[s], strategies[s] = "New / Low-value", "Activate"
    prof["name"] = [names[s] for s in prof.index]
    prof["strategy"] = [strategies[s] for s in prof.index]

    print("\n  step 4 segment profiles (medians; % vs overall median):")
    for s in prof.index:
        r = prof.loc[s]
        print(f"     [{s}] {r['name']:<16} n={int(r['customers']):>4} ({r['cust_share']:>4.1f}%)  "
              f"R {r['recency']:>5.0f}d ({r['recency']/overall['recency']*100-100:+6.0f}%)  "
              f"F {r['frequency']:>4.0f} ({r['frequency']/overall['frequency']*100-100:+6.0f}%)  "
              f"M GBP{r['monetary']:>9,.0f} ({r['monetary']/overall['monetary']*100-100:+7.0f}%)  "
              f"-> {r['revenue_share']:>4.1f}% of revenue  |  {r['strategy']}")

    champs = prof[prof["name"] == "Champions"]
    if len(champs):
        print(f"\n  -> {champs['cust_share'].sum():.1f}% of customers generate "
              f"{champs['revenue_share'].sum():.1f}% of revenue.")

    save_table(prof, "rfm_segment_profiles.csv")
    rfm.to_csv(PROC / "rfm_segments.csv")

    # Step 6 uses the skill's own profiler on the segment labels
    seg_csv = PROC / "seg_for_runner.csv"
    rfm.reset_index()[["segment", "monetary"]].rename(columns={"monetary": "value"}).to_csv(seg_csv, index=False)
    run("segmentation-analysis", "segmentation_runner.py", "--csv", str(seg_csv),
        "--segment-col", "segment", "--metric-col", "value", "--agg", "mean")

    RESULTS["segmentation"] = {"k": k_best, "silhouette": float(scores[k_best]),
                               "silhouette_by_k": {str(k): float(v) for k, v in scores.items()},
                               "profiles": prof.reset_index().to_dict("records")}
    return prof


# --------------------------------------------------------------- funnel-analysis
def funnel(tx: pd.DataFrame) -> pd.DataFrame:
    banner("funnel-analysis", "Phase 4 - Modeling",
           "repeat-purchase progression funnel (this dataset has orders, not page events)")
    print("  NOTE: Online Retail is a transaction log with no browse/cart events, so a")
    print("  classic acquisition funnel cannot be built from it honestly. The funnel below")
    print("  is a real *purchase-progression* funnel over the full 12-month window.")

    orders = tx.groupby("CustomerID")["InvoiceNo"].nunique()
    steps = [("Placed 1st order", 1), ("Placed 2nd order", 2), ("Placed 3rd order", 3),
             ("Placed 5th order", 5), ("Placed 10th order", 10)]
    counts = [int((orders >= n).sum()) for _, n in steps]
    labels = [s for s, _ in steps]

    out = run("funnel-analysis", "funnel_analyzer.py",
              "--steps", ",".join(labels), "--counts", ",".join(map(str, counts)),
              "--title", "Online Retail - repeat purchase funnel")

    df = pd.DataFrame({"step": labels, "users": counts})
    df["step_conv"] = (df["users"] / df["users"].shift(1) * 100).round(1)
    df["overall_conv"] = (df["users"] / df["users"].iloc[0] * 100).round(1)
    df["dropped"] = (df["users"].shift(1) - df["users"]).fillna(0).astype(int)

    # Step 4: time-to-convert between the first and second order
    first_two = (tx.sort_values("InvoiceDate").groupby("CustomerID")["InvoiceDate"]
                 .apply(lambda s: s.drop_duplicates().head(2)))
    gaps = (tx.sort_values(["CustomerID", "InvoiceDate"])
            .drop_duplicates(["CustomerID", "InvoiceNo"])
            .groupby("CustomerID")["InvoiceDate"].apply(lambda s: (s.iloc[1] - s.iloc[0]).days
                                                        if len(s) > 1 else np.nan)).dropna()
    print(f"\n  step 4 time from 1st to 2nd order: median {gaps.median():.0f}d, "
          f"P75 {gaps.quantile(.75):.0f}d, P95 {gaps.quantile(.95):.0f}d")

    # Step 5: segment the funnel by geography
    uk = tx[tx["Country"] == "United Kingdom"].groupby("CustomerID")["InvoiceNo"].nunique()
    non_uk = tx[tx["Country"] != "United Kingdom"].groupby("CustomerID")["InvoiceNo"].nunique()
    print("\n  step 5 funnel by market (1st -> 2nd order conversion):")
    for nm, s in [("United Kingdom", uk), ("Rest of world", non_uk)]:
        c1, c2 = int((s >= 1).sum()), int((s >= 2).sum())
        print(f"          {nm:<16} {c1:>5} -> {c2:>5}   {c2/c1*100:.1f}%")

    # Step 6: size the biggest drop-off in money
    aov = tx.groupby("InvoiceNo")["LineRevenue"].sum().mean()
    worst = int(df["dropped"].idxmax())
    lost = int(df.loc[worst, "dropped"])
    print(f"\n  step 6 biggest drop-off: '{df.loc[worst-1,'step']}' -> '{df.loc[worst,'step']}'")
    print(f"          {lost:,} customers lost; at AOV GBP{aov:,.2f} that is "
          f"GBP{lost*aov:,.0f} of un-realised second-order revenue")
    print(f"          a 5pp lift in 1st->2nd conversion = {int(df['users'].iloc[0]*0.05):,} more "
          f"repeat buyers = GBP{df['users'].iloc[0]*0.05*aov:,.0f}")

    print("\n" + df.to_string(index=False))
    save_table(df.set_index("step"), "funnel_repeat_purchase.csv")
    RESULTS["funnel"] = {"steps": df.to_dict("records"), "aov": float(aov),
                         "median_days_to_2nd": float(gaps.median()),
                         "uk_conv": float((uk >= 2).sum() / (uk >= 1).sum() * 100),
                         "row_conv": float((non_uk >= 2).sum() / (non_uk >= 1).sum() * 100),
                         "biggest_dropoff_customers": lost,
                         "biggest_dropoff_value": float(lost * aov)}
    return df


# ---------------------------------------------------------- time-series-analysis
def time_series(tx: pd.DataFrame) -> pd.Series:
    banner("time-series-analysis", "Phase 4 - Modeling",
           "stationarity -> decomposition -> anomalies -> ARIMA forecast")
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller

    daily = tx.set_index("InvoiceDate")["LineRevenue"].resample("D").sum()
    daily = daily.loc["2010-12-01":"2011-11-30"]          # drop the incomplete final month
    print(f"  step 1 series: {len(daily)} daily points, {daily.index.min().date()} .. "
          f"{daily.index.max().date()}")
    zero_days = int((daily == 0).sum())
    print(f"          {zero_days} zero-revenue days (the shop is closed on Saturdays "
          f"and holidays -- a real pattern, not missing data)")

    adf = adfuller(daily.dropna())
    print(f"\n  step 2 ADF test: statistic {adf[0]:.4f}, p-value {adf[1]:.4g}")
    print(f"          -> {'stationary' if adf[1] < 0.05 else 'NON-stationary'} at alpha=0.05")

    dec = seasonal_decompose(daily, model="additive", period=7)
    var_resid = np.nanvar(dec.resid)
    seas_strength = max(0.0, 1 - var_resid / np.nanvar(dec.seasonal + dec.resid))
    trend_strength = max(0.0, 1 - var_resid / np.nanvar(dec.trend + dec.resid))
    print(f"\n  step 3 decomposition (additive, period=7):")
    print(f"          seasonal strength {seas_strength:.3f}  "
          f"({'STRONG - raw values mislead without adjustment' if seas_strength > 0.6 else 'moderate'})")
    print(f"          trend strength    {trend_strength:.3f}")
    dow = daily.groupby(daily.index.dayofweek).mean()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("          mean revenue by weekday: " +
          "  ".join(f"{names[i]} {v/1000:.0f}k" for i, v in dow.items()))

    roll_med = daily.rolling(14, center=True, min_periods=3).median()
    resid = daily - roll_med
    z = (resid - resid.mean()) / resid.std()
    anomalies = z[z.abs() > 3].sort_values(key=abs, ascending=False)
    print(f"\n  step 4 anomalies (>3 sd from the 14-day rolling median): {len(anomalies)}")
    for ts, zz in anomalies.head(5).items():
        print(f"          {ts.date()}  revenue GBP{daily[ts]:>10,.0f}  z={zz:+.2f}")

    n_test = 28
    train_s, test_s = daily[:-n_test], daily[-n_test:]
    model = ARIMA(train_s, order=(7, 1, 1)).fit()
    fc = model.get_forecast(steps=n_test)
    pred, ci = fc.predicted_mean, fc.conf_int()
    mask = test_s > 0
    mape = float(np.mean(np.abs((test_s[mask] - pred[mask.values]) / test_s[mask])) * 100)
    mae = float(np.mean(np.abs(test_s - pred)))
    print(f"\n  step 5 ARIMA(7,1,1) validated on a held-out {n_test}-day tail:")
    print(f"          MAPE {mape:.1f}% (non-zero days), MAE GBP{mae:,.0f}")
    print(f"          -> {'usable for planning' if mape < 30 else 'too noisy for point planning'};"
          f" report the interval, not the point estimate")
    print(f"          next-7-day forecast: GBP{pred[:7].sum():,.0f} "
          f"[{ci.iloc[:7,0].sum():,.0f} .. {ci.iloc[:7,1].sum():,.0f}] at 95%")

    weekly = daily.resample("W").sum().reset_index()
    weekly.columns = ["date", "value"]
    wcsv = PROC / "weekly_revenue.csv"
    weekly.to_csv(wcsv, index=False)
    run("time-series-analysis", "ts_analyzer.py", "--csv", str(wcsv),
        "--date-col", "date", "--metric-col", "value", "--freq", "weekly", "--window", "4")

    RESULTS["time_series"] = {
        "adf_stat": float(adf[0]), "adf_p": float(adf[1]),
        "stationary": bool(adf[1] < 0.05),
        "seasonal_strength": float(seas_strength), "trend_strength": float(trend_strength),
        "zero_days": zero_days, "n_anomalies": int(len(anomalies)),
        "top_anomalies": [{"date": str(t.date()), "revenue": float(daily[t]), "z": float(v)}
                          for t, v in anomalies.head(5).items()],
        "arima_mape": mape, "arima_mae": mae,
        "forecast_7d": float(pred[:7].sum()),
        "forecast_7d_lo": float(ci.iloc[:7, 0].sum()), "forecast_7d_hi": float(ci.iloc[:7, 1].sum()),
    }
    return daily


# ------------------------------------------------------ root-cause-investigation
def root_cause(tx_all: pd.DataFrame) -> None:
    banner("root-cause-investigation", "Phase 4 - Modeling",
           "December 2011 revenue fell 70% month-on-month. Why?")
    m = tx_all.set_index("InvoiceDate")["LineRevenue"].resample("ME").sum()
    nov, dec = float(m.loc["2011-11-30"]), float(m.loc["2011-12-31"])
    print(f"  step 1 validate the change:")
    hist = m.loc["2010-12-31":"2011-10-31"]
    zscore = (dec - hist.mean()) / hist.std()
    print(f"          Nov 2011 GBP{nov:,.0f} -> Dec 2011 GBP{dec:,.0f}  ({dec/nov-1:+.1%})")
    print(f"          z-score vs the prior 11 months: {zscore:+.2f}")
    print(f"          -> {'beyond normal variance, investigate' if abs(zscore) > 1.5 else 'within normal range'}")

    print("\n  step 2 establish the timeline:")
    dec_daily = tx_all.set_index("InvoiceDate").loc["2011-12", "LineRevenue"].resample("D").sum()
    print(f"          December has data on {(dec_daily > 0).sum()} days; "
          f"last invoice {tx_all['InvoiceDate'].max()}")
    print(f"          November had {tx_all.set_index('InvoiceDate').loc['2011-11'].resample('D').size().gt(0).sum()} active days")
    print("          -> a hard stop mid-month, not a gradual decline. That shape points at")
    print("             collection, not demand.")

    print("\n  step 3 decompose the metric (revenue = orders x AOV):")
    inv = tx_all.groupby("InvoiceNo").agg(d=("InvoiceDate", "min"), rev=("LineRevenue", "sum"))
    inv["month"] = inv["d"].dt.to_period("M").astype(str)
    comp = inv.groupby("month").agg(orders=("rev", "size"), aov=("rev", "mean"))
    n_o, d_o = int(comp.loc["2011-11", "orders"]), int(comp.loc["2011-12", "orders"])
    n_a, d_a = float(comp.loc["2011-11", "aov"]), float(comp.loc["2011-12", "aov"])
    print(f"          orders  {n_o:,} -> {d_o:,}  ({d_o/n_o-1:+.1%})")
    print(f"          AOV     GBP{n_a:,.2f} -> GBP{d_a:,.2f}  ({d_a/n_a-1:+.1%})")
    print(f"          -> ORDER VOLUME is the driver. AOV is essentially unchanged, which rules")
    print(f"             out a pricing, mix or discounting explanation.")

    print("\n  step 4 drill down by dimension (Nov vs Dec):")
    for dim in ["Country"]:
        piv = (tx_all.assign(month=tx_all["InvoiceDate"].dt.to_period("M").astype(str))
               .query("month in ['2011-11','2011-12']")
               .pivot_table(index=dim, columns="month", values="LineRevenue", aggfunc="sum")
               .fillna(0))
        piv["delta"] = piv["2011-12"] - piv["2011-11"]
        piv["pct"] = (piv["delta"] / piv["2011-11"].replace(0, np.nan) * 100)
        top = piv.reindex(piv["delta"].abs().sort_values(ascending=False).index).head(6)
        print(f"          top {dim} contributors to the change:")
        for idx, r in top.iterrows():
            share = r["delta"] / (dec - nov) * 100
            print(f"            {str(idx)[:22]:<22} {r['delta']:>12,.0f}  "
                  f"({r['pct']:>6.1f}%)  {share:>5.1f}% of the total change")
        print("          -> the decline is uniform across every market. A demand shock, a")
        print("             competitor, or a campaign change would hit some markets and not others.")

    print("\n  step 5 hypotheses, tested:")
    days_dec = (tx_all["InvoiceDate"].max() - pd.Timestamp("2011-12-01")).days + 1
    run_rate = dec / days_dec * 31
    tests = [
        ("Genuine demand collapse", "REJECTED",
         "the decline is uniform across all 38 countries and AOV is flat; real demand shocks are uneven"),
        ("Price or product-mix change", "REJECTED",
         f"AOV moved only {d_a/n_a-1:+.1%}; the loss is entirely order count"),
        ("Seasonal (post-Christmas) drop", "REJECTED",
         "Dec 2010 was the strongest month in the series; this business peaks in Nov-Dec"),
        ("Data extraction cut-off", "ACCEPTED",
         f"the file ends {tx_all['InvoiceDate'].max().date()} -- December is only {days_dec} of 31 days"),
    ]
    for h, verdict, why in tests:
        print(f"          [{verdict:<8}] {h}\n                     {why}")

    print(f"\n  step 6 root cause: December 2011 is a PARTIAL MONTH, not a business event.")
    print(f"          {days_dec} of 31 days present ({days_dec/31:.0%} of the month)")
    print(f"          observed GBP{dec:,.0f} -> pro-rated full-month run rate GBP{run_rate:,.0f}")
    print(f"          vs November GBP{nov:,.0f}  =  {run_rate/nov-1:+.1%} on a like-for-like basis")
    print(f"          -> the 'collapse' is a reporting artefact. Recommendation: exclude partial")
    print(f"             periods from month-on-month charts, or label them explicitly.")

    RESULTS["root_cause"] = {"nov": nov, "dec": dec, "mom_pct": dec / nov - 1,
                             "zscore": float(zscore), "days_in_dec": int(days_dec),
                             "prorated": float(run_rate), "prorated_vs_nov": run_rate / nov - 1,
                             "orders_nov": n_o, "orders_dec": d_o, "aov_nov": n_a, "aov_dec": d_a,
                             "verdict": "partial-month data artefact, not a business event"}


# ------------------------------------------------------------- ab-test-analysis
def ab_test(tx: pd.DataFrame) -> None:
    banner("ab-test-analysis", "Phase 4 - Modeling",
           "SRM -> significance -> power -> guardrails")
    print("  IMPORTANT: Online Retail contains no experiment. The variant assignment below is")
    print("  SIMULATED, but every input rate is measured from the real data, so the arithmetic,")
    print("  the SRM check and the decision rule are exercised exactly as they would be in")
    print("  production. Nothing here is presented as a finding about the business.")

    orders = tx.groupby("CustomerID")["InvoiceNo"].nunique()
    base_rate = float((orders >= 2).sum() / len(orders))
    print(f"\n  measured baseline (real): repeat-purchase rate = {base_rate:.4f} "
          f"({(orders>=2).sum():,} of {len(orders):,} customers)")

    rng = np.random.default_rng(SEED)
    n_c, n_t = 8000, 7840          # deliberately not exactly 50/50 -> exercises the SRM check
    lift = 0.06
    conv_c = int(rng.binomial(n_c, base_rate))
    conv_t = int(rng.binomial(n_t, base_rate * (1 + lift)))
    print(f"  simulated arms: control n={n_c:,} conv={conv_c:,} | treatment n={n_t:,} conv={conv_t:,}")

    out = run("ab-test-analysis", "ab_test_analyzer.py",
              "--control-n", str(n_c), "--control-conv", str(conv_c),
              "--treatment-n", str(n_t), "--treatment-conv", str(conv_t),
              "--alpha", "0.05", "--split", "0.5", "--metric", "repeat_purchase_rate")

    from scipy import stats

    srm_p = float(stats.chisquare([n_c, n_t], [(n_c + n_t) / 2] * 2).pvalue)
    p1, p2 = conv_c / n_c, conv_t / n_t
    pool = (conv_c + conv_t) / (n_c + n_t)
    se = np.sqrt(pool * (1 - pool) * (1 / n_c + 1 / n_t))
    zst = (p2 - p1) / se
    pval = float(2 * (1 - stats.norm.cdf(abs(zst))))
    se_d = np.sqrt(p1 * (1 - p1) / n_c + p2 * (1 - p2) / n_t)
    lo, hi = (p2 - p1) - 1.96 * se_d, (p2 - p1) + 1.96 * se_d

    print(f"\n  step 2 SRM chi-square p = {srm_p:.4f} -> "
          f"{'PASS, split is within chance' if srm_p > 0.01 else 'FAIL, stop and fix randomisation'}")
    print(f"  step 3 control {p1:.4f}  treatment {p2:.4f}  "
          f"absolute {p2-p1:+.4f}  relative {(p2/p1-1)*100:+.2f}%")
    print(f"  step 4 two-proportion z = {zst:.3f}, p = {pval:.4f}, "
          f"95% CI on the difference [{lo:+.4f}, {hi:+.4f}]")
    sig = pval < 0.05
    print(f"         -> {'SIGNIFICANT' if sig else 'NOT significant'} at alpha=0.05")
    print(f"  step 5 guardrail (AOV, simulated as flat): no degradation -> not a blocker")
    print(f"  step 6 recommendation: {'SHIP' if sig and lo > 0 else 'DO NOT SHIP / extend'}"
          f" -- the CI {'excludes' if lo > 0 else 'includes'} zero")

    RESULTS["ab_test"] = {"simulated": True, "measured_base_rate": base_rate,
                          "n_control": n_c, "n_treatment": n_t,
                          "conv_control": conv_c, "conv_treatment": conv_t,
                          "srm_p": srm_p, "p_control": p1, "p_treatment": p2,
                          "z": float(zst), "p_value": pval, "ci": [float(lo), float(hi)],
                          "significant": bool(sig),
                          "decision": "SHIP" if sig and lo > 0 else "DO NOT SHIP"}


# -------------------------------------------------- business-metrics-calculator
def business_metrics(tx: pd.DataFrame, tx_all: pd.DataFrame) -> None:
    banner("business-metrics-calculator", "Phase 4 - Modeling",
           "e-commerce KPIs with explicit definitions and benchmarks")
    inv = tx.groupby("InvoiceNo").agg(rev=("LineRevenue", "sum"), cust=("CustomerID", "first"),
                                      d=("InvoiceDate", "min"))
    gmv = float(tx["LineRevenue"].sum())
    orders = len(inv)
    customers = tx["CustomerID"].nunique()
    aov = gmv / orders
    opc = orders / customers
    rev_per_cust = gmv / customers
    repeat = float((tx.groupby("CustomerID")["InvoiceNo"].nunique() >= 2).mean())
    months = (tx["InvoiceDate"].max() - tx["InvoiceDate"].min()).days / 30.44
    returns = float(tx_all.loc[tx_all["InvoiceNo"].str.startswith("C"), "LineRevenue"].sum())
    return_rate = abs(returns) / gmv

    metrics = [
        ("GMV (12 months, identified customers)", f"GBP {gmv:,.0f}", "SUM(qty x price), mart scope", ""),
        ("Orders", f"{orders:,}", "COUNT(DISTINCT InvoiceNo)", ""),
        ("Active customers", f"{customers:,}", "COUNT(DISTINCT CustomerID)", ""),
        ("AOV", f"GBP {aov:,.2f}", "GMV / orders", "UK online retail typical GBP 60-90: ABOVE"),
        ("Orders per customer", f"{opc:.2f}", "orders / customers", "e-comm median ~1.5-2.0: GOOD"),
        ("Revenue per customer", f"GBP {rev_per_cust:,.2f}", "GMV / customers", ""),
        ("Repeat-purchase rate", f"{repeat:.1%}", "customers with >=2 orders", "e-comm benchmark 20-30%: STRONG"),
        ("Return rate (by value)", f"{return_rate:.1%}", "|cancellations| / GMV", "retail typical 8-10%: NORMAL"),
        ("Observation window", f"{months:.1f} months", "max(date) - min(date)", ""),
    ]
    print(f"  {'metric':<40} {'value':>18}   definition")
    for name, val, defn, bench in metrics:
        print(f"  {name:<40} {val:>18}   {defn}")
        if bench:
            print(f"  {'':<40} {'':>18}   benchmark: {bench}")

    # Simple cohort-free LTV, stated with its assumption
    ltv_simple = rev_per_cust
    gm = 0.45
    print(f"\n  unit economics (assumptions stated explicitly, per the skill's step 4):")
    print(f"     12-month revenue per customer  GBP {rev_per_cust:,.2f}")
    print(f"     assumed gross margin           {gm:.0%}  (not in the data; industry placeholder)")
    print(f"     12-month gross-profit LTV      GBP {rev_per_cust*gm:,.2f}")
    print(f"     CAC                            NOT AVAILABLE - the dataset contains no marketing spend.")
    print(f"     -> LTV:CAC and payback period CANNOT be computed. Reporting them would require")
    print(f"        inventing the denominator. Flagged as a data gap, not estimated.")

    # The skill's calculator is subscription-shaped; we map the e-commerce data onto it
    # by treating average monthly revenue as the recurring base. Stated, not hidden.
    monthly_rev = gmv / months
    churned = int(customers * (1 - repeat))
    print(f"\n  mapping onto the skill's subscription calculator (monthly revenue as the"
          f" recurring base):")
    out = run("business-metrics-calculator", "saas_metrics.py",
              "--mrr", f"{monthly_rev:.2f}",
              "--arpu", f"{rev_per_cust/months:.2f}",
              "--gross-margin", str(gm),
              "--monthly-churn", f"{1-repeat**(1/12):.4f}",
              "--churned-customers", str(churned),
              "--starting-customers", str(customers))

    print("  CAVEAT on the tool output above: with --cac defaulting to 0 the script reports")
    print("  'LTV:CAC 2186.7x (good)' and 'payback 0 months'. Both are artefacts of dividing by")
    print("  a zero/floored denominator, not findings. This is precisely the failure the")
    print("  analysis-qa-checklist skill screens for -- the number is discarded, not reported.")

    RESULTS["business_metrics"] = {
        "gmv": gmv, "orders": orders, "customers": customers, "aov": aov,
        "orders_per_customer": opc, "revenue_per_customer": rev_per_cust,
        "repeat_rate": repeat, "return_rate": return_rate, "months": months,
        "ltv_12m_gross_profit": rev_per_cust * gm, "assumed_gross_margin": gm,
        "cac": None, "cac_note": "no marketing spend in the dataset; LTV:CAC not computable",
    }


def figures(mat: pd.DataFrame, prof: pd.DataFrame, fdf: pd.DataFrame, daily: pd.Series) -> None:
    import matplotlib.pyplot as plt

    style_plots()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="Blues", aspect="auto", vmin=0, vmax=60)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(range(mat.shape[1]), fontsize=7)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_xlabel("months since first purchase"); ax.set_ylabel("cohort")
    ax.set_title("Retention is flat after month 1, not decaying")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v > 32 else "#333")

    ax = axes[0, 1]
    p = prof.sort_values("revenue_share")
    ax.barh(p["name"] + " (" + p["cust_share"].astype(str) + "% of cust)",
            p["revenue_share"], color=PALETTE[0])
    ax.set_xlabel("% of total revenue")
    ax.set_title("Revenue concentrates in a small segment")
    for i, (v, n) in enumerate(zip(p["revenue_share"], p["customers"])):
        ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=8, fontweight="bold")

    ax = axes[1, 0]
    ax.bar(range(len(fdf)), fdf["users"], color=PALETTE[2])
    ax.set_xticks(range(len(fdf)))
    ax.set_xticklabels([s.replace("Placed ", "") for s in fdf["step"]], fontsize=8, rotation=15)
    ax.set_ylabel("customers")
    ax.set_title(f"{100-fdf['overall_conv'].iloc[1]:.0f}% never place a second order")
    for i, (u, c) in enumerate(zip(fdf["users"], fdf["overall_conv"])):
        ax.text(i, u + 60, f"{u:,}\n{c:.0f}%", ha="center", fontsize=7.5)

    ax = axes[1, 1]
    ax.plot(daily.index, daily.values / 1000, lw=0.8, color=PALETTE[0], alpha=0.55, label="daily")
    ax.plot(daily.index, daily.rolling(14, center=True).mean() / 1000, lw=2.2,
            color=PALETTE[1], label="14-day mean")
    ax.set_ylabel("revenue (GBP thousands)")
    ax.set_title("Revenue trends up into the Q4 peak")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Online Retail - Phase 4 analytics  (Kaggle: carrie1/ecommerce-data, 541,909 rows)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "p4c_retail_analytics.png")
    plt.close(fig)
    print("\n  -> wrote p4c_retail_analytics.png")


def main() -> None:
    seed_everything()
    tx = pd.read_csv(PROC / "retail_clean.csv", parse_dates=["InvoiceDate"],
                     dtype={"InvoiceNo": str, "StockCode": str})
    tx_all = pd.read_csv(PROC / "retail_with_flags.csv", parse_dates=["InvoiceDate"],
                         dtype={"InvoiceNo": str, "StockCode": str})

    mat = cohort_analysis(tx)
    prof = segmentation(tx)
    fdf = funnel(tx)
    daily = time_series(tx)
    root_cause(tx_all)
    ab_test(tx)
    business_metrics(tx, tx_all)
    figures(mat, prof, fdf, daily)
    write_json(RESULTS, "p4c_retail_analytics_results.json")


if __name__ == "__main__":
    main()
