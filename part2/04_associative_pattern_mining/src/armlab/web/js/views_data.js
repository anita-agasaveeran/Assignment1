/* Views for CRISP-DM phases 1-3: business framing, data understanding, preparation. */
import * as C from "./charts.js";
import {A} from "./api.js";
import {h, esc, tile, card, table, kv, dot, chip, note, cite, grid, section, inlineBar} from "./ui.js";

const f = C.fmt;

const PHASES = [
  ["1", "Business understanding", "Frame the merchandising question and the success criteria."],
  ["2", "Data understanding", "Profile 541,909 transaction lines; quantify what is broken."],
  ["3", "Data preparation", "Clean, de-duplicate, build baskets, split on time."],
  ["4", "Modeling", "FP-Growth + rule generation, tuned by AutoResearch hill climbing."],
  ["5", "Evaluation", "Holdout replication, exact tests, FDR control, null baseline."],
  ["6", "Deployment", "Model card, REST API, live recommender."],
];

export async function overview(el) {
  section(el, "");
  const [run, prof, ev, rules, its] = await Promise.all([
    A.run(), A.profile(), A.evaluation(), A.rules({limit: 8, sort: "leverage"}), A.itemsets()]);
  const ho = ev.holdout, nb = ev.null_baseline || {};
  const base = ev.baseline.objective.value, champ = ev.champion.objective.value;

  el.appendChild(h(`<div class="view-head">
    <p>Market-basket association mining on the <b>Online Retail</b> transaction log
    (Kaggle <span class="mono">carrie1/ecommerce-data</span>; UCI ML Repository 352).
    Run end to end as CRISP-DM, with hyperparameters chosen by an automated
    hill-climbing search against a holdout-validated objective rather than by hand.</p></div>`));

  el.appendChild(grid("g4", [
    tile({label: "Baskets analysed", value: f.n(run.preparation.n_transactions),
      foot: `<b>${f.n(run.preparation.n_items)}</b> distinct products · avg basket
             <b>${f.f(run.preparation.avg_basket_size, 1)}</b> items`}),
    tile({label: "Rules discovered", value: f.n(run.n_rules),
      foot: `from <b>${f.n(its.n_itemsets)}</b> frequent itemsets at support ≥ <b>${f.pct(its.min_support, 2)}</b>`}),
    tile({label: "Holdout-validated", value: f.n(ho.n_validated),
      foot: `<b>${f.pct(ho.replication_rate)}</b> of rules replicated on unseen future baskets`}),
    tile({label: "Holdout coverage", value: f.pct(ho.coverage, 1),
      foot: `share of future baskets where at least one rule fires`}),
  ]));

  el.appendChild(h(`<div class="phase-rail">${PHASES.map(([n, t, d]) =>
    `<div class="p done"><div class="n">PHASE ${n}</div><div class="t">${esc(t)}</div>
     <div class="d">${esc(d)}</div></div>`).join("")}</div>`));

  const c1 = card({title: "Did the automated search actually help?",
    sub: "AutoResearch objective decomposition: the hand-set baseline configuration against the "
       + "champion the hill climber found. Each bar is the weighted contribution of one term.",
    right: chip(`+${f.pct((champ - base) / base, 1)} objective`, champ > base ? "good" : "")});
  el.appendChild(c1);
  const terms = Object.keys(ev.champion.objective.weighted);
  const cols = C.SERIES();
  C.stacked(c1.body, {
    categories: ["Baseline (hand-set)", "Champion (AutoResearch)"],
    series: terms.map((t, i) => ({name: t.replace(/_/g, " "), color: cols[i],
      values: [ev.baseline.objective.weighted[t], ev.champion.objective.weighted[t]]})),
    yfmt: (v) => f.f(v, 3), yLabel: "weighted objective contribution",
    tipRows: (s, cat, v) => [["term", s.name], ["weighted", f.f(v, 4)]],
  });

  const c2 = card({title: "Headline findings",
    sub: "The numbers a reviewer should check first."});
  el.appendChild(c2);
  c2.body.appendChild(table({
    cols: [{label: "Check", wrap: true}, {label: "Result", num: true}, {label: "Reading", wrap: true}],
    raw: true,
    rows: [
      [`${dot("good")} Holdout replication`, `<b>${f.pct(ho.replication_rate)}</b>`,
       `${f.n(ho.n_replicated)} of ${f.n(ho.n_rules)} rules still clear their thresholds on
        transactions mined after the training window closed.`],
      [`${dot(ho.lift_rank_spearman > .7 ? "good" : "warn")} Lift rank stability`,
       `ρ = <b>${f.f(ho.lift_rank_spearman, 3)}</b>`,
       `Spearman correlation between in-sample and holdout lift. Rules keep their
        relative order out of sample, which is what a ranked recommender needs.`],
      [`${dot("good")} Confidence drift`, `<b>${f.f(ho.confidence_drift_mean, 4)}</b>`,
       `Mean confidence change from explore to holdout. Slightly negative is the
        expected optimism of threshold selection; a large drop would signal overfitting.`],
      [`${dot(nb.n_rules_null === 0 ? "good" : "warn")} Swap-randomised null`,
       `<b>${f.n(nb.n_rules_null ?? 0)}</b> rules`,
       `Mining the identical thresholds on data with the same item frequencies and
        basket sizes but destroyed co-occurrence structure
        (${f.compact(nb.n_swaps_performed ?? 0)} swaps) yields
        ${nb.n_rules_null === 0 ? "<b>nothing at all</b>" : f.n(nb.n_rules_null) + " rules"} —
        an empirical false-discovery rate of ${f.f(nb.empirical_fdr ?? 0, 3)}.`],
      [`${dot("good")} Multiplicity control`,
       `<b>${f.n(ev.significance?.n_significant_corrected ?? 0)}</b> significant`,
       `Fisher exact p-values corrected against the
        <b>${f.compact(ev.significance?.n_hypotheses_corrected_for ?? 0)}</b>-hypothesis
        search space, not just the reported rules.`],
      [`${dot(ev.benchmark.all_agree ? "good" : "bad")} Algorithm cross-check`,
       ev.benchmark.all_agree ? "<b>identical</b>" : "<b>DIVERGED</b>",
       `FP-Growth, Apriori and Eclat were all implemented from scratch and return
        byte-identical itemset→count maps at the same threshold.`],
    ]}));

  const c3 = card({title: "Strongest rules by leverage",
    sub: "Leverage (Piatetsky-Shapiro 1991) measures excess co-occurrence over independence — "
       + "it favours rules that are both surprising and frequent enough to matter commercially.",
    right: `<a href="#rules" class="chip">open rule explorer →</a>`});
  el.appendChild(c3);
  c3.body.appendChild(table({raw: true,
    cols: [{label: "Rule", wrap: true}, {label: "Support", num: true}, {label: "Conf", num: true},
           {label: "Lift", num: true}, {label: "Leverage", num: true}, {label: "Holdout", num: true}],
    rows: rules.rules.map(r => [
      ruleHtml(r), f.pct(r.support, 2), f.pct(r.confidence, 1), f.f(r.lift, 2),
      f.f(r.leverage, 4),
      r.validated ? `${dot("good")} held` : `${dot("warn")} drifted`])}));
}

export function ruleHtml(r) {
  return `<span class="rule-cell"><span class="ante">${esc(r.antecedent_items.join(" + "))}</span>
    <span class="arrow">⇒</span><span class="cons">${esc(r.consequent_items.join(" + "))}</span></span>`;
}

/* ------------------------------------------------------------------ */
export async function business(el) {
  section(el, "");
  const [prof, mc] = await Promise.all([A.profile(), A.modelCard()]);
  el.appendChild(h(`<div class="view-head">
    <p>CRISP-DM Phase 1. Everything downstream — the cleaning rules, the split, the
    objective the optimiser maximises — is a consequence of the decisions on this page.
    Stated first so they can be argued with.</p></div>`));

  const c1 = card({title: "The business question"});
  el.appendChild(c1);
  c1.body.innerHTML = `
    <p style="margin-top:0">A UK-based online giftware retailer wants to grow
    <b>basket size</b>. Three concrete decisions depend on knowing which products are
    bought together:</p>
    <ol style="color:var(--text-secondary);padding-left:20px">
      <li><b>Cross-sell surface.</b> What to show on the product page and in the cart.</li>
      <li><b>Bundles.</b> Which pairs or triples to package at a discount.</li>
      <li><b>Catalogue adjacency.</b> How to order the printed catalogue and category pages.</li>
    </ol>
    ${note(`<b>Why association rules rather than a supervised model?</b> There is no
      label. The question is not "will this customer convert" but "what structure
      exists in the co-purchase data" — an unsupervised, descriptive task whose output
      has to be legible to a merchandiser, not just accurate. A rule is a sentence a
      human can act on; an embedding is not.`)}`;

  const c2 = card({title: "Success criteria",
    sub: "Data-mining goals, stated as thresholds before the modelling starts, so the "
       + "objective function is not reverse-engineered from whatever the model produced."});
  el.appendChild(c2);
  c2.body.appendChild(table({raw: true,
    cols: [{label: "Criterion", wrap: true}, {label: "Target", num: true},
           {label: "Achieved", num: true}, {label: "Why this bar", wrap: true}],
    rows: [
      ["Rules replicate on unseen data", "≥ 60%",
       `<b>${f.pct(mc.metrics.replication_rate)}</b> ${dot("good")}`,
       "A rule that does not survive a temporal holdout is a description of last quarter, not a decision rule."],
      ["Rule set fires on real baskets", "≥ 40%",
       `<b>${f.pct(mc.metrics.holdout_coverage)}</b> ${dot("good")}`,
       "A cross-sell surface that is blank on most product pages is not shippable."],
      ["Survives multiplicity correction", "> 0 rules",
       `<b>${f.n(mc.metrics.n_holdout_validated)}</b> ${dot("good")}`,
       "Mining evaluates trillions of hypotheses; uncorrected significance is meaningless."],
      ["Beats a structure-free null", "FDR < 0.05",
       `<b>${f.f(mc.metrics.empirical_fdr_vs_null ?? 0, 3)}</b> ${dot("good")}`,
       "The distribution-free check that the patterns are not an artefact of item popularity."],
      ["Human-legible output", "≤ 3 antecedent items",
       `<b>${mc.model_details.hyperparameters.max_antecedent_len}</b> ${dot("good")}`,
       "A merchandiser will not act on a seven-item conjunction."],
    ]}));

  const c3 = card({title: "Out of scope — stated up front",
    sub: "The failure mode of a market-basket project is a stakeholder reading causation into it."});
  el.appendChild(c3);
  c3.body.innerHTML = `<ul style="color:var(--text-secondary);padding-left:20px;margin:0">
    ${mc.intended_use.out_of_scope.map(s => `<li>${esc(s)}</li>`).join("")}</ul>`;

  const c4 = card({title: "Data inventory", sub: "What the raw source actually contains."});
  el.appendChild(c4);
  c4.body.appendChild(kv([
    ["Source", `Online Retail — UCI ML Repository 352 / Kaggle <span class="mono">carrie1/ecommerce-data</span>`],
    ["Provenance", "Chen, Sain &amp; Guo (2012), <i>J. Database Marketing</i> 19:197–208"],
    ["Grain", "one row per invoice line item"],
    ["Rows", f.n(prof.n_rows)],
    ["Period", `${prof.date_min.slice(0, 10)} → ${prof.date_max.slice(0, 10)} (${prof.span_days} days)`],
    ["Invoices", f.n(prof.n_invoices)],
    ["Products", f.n(prof.n_items)],
    ["Customers", f.n(prof.n_customers)],
    ["Countries", f.n(prof.n_countries)],
    ["Gross revenue", "£" + f.compact(prof.gross_revenue)],
  ]));
}

/* ------------------------------------------------------------------ */
export async function data(el) {
  section(el, "");
  const prof = await A.profile();
  el.appendChild(h(`<div class="view-head">
    <p>CRISP-DM Phase 2. Descriptive only — nothing is dropped on this page. The
    point is to <em>quantify</em> what is wrong before deciding what to do about it.</p></div>`));

  el.appendChild(grid("g4", [
    tile({label: "Invoice lines", value: f.n(prof.n_rows), foot: `${prof.n_columns} columns`}),
    tile({label: "Invoices", value: f.n(prof.n_invoices),
      foot: `median basket <b>${f.n(prof.basket_size.median)}</b> items,
             <b>${f.n(prof.basket_size.n_singleton)}</b> singletons`}),
    tile({label: "Products", value: f.n(prof.n_items),
      foot: `across <b>${f.n(prof.n_countries)}</b> countries`}),
    tile({label: "Observation window", value: prof.span_days, unit: "days",
      foot: `${prof.date_min.slice(0, 10)} → ${prof.date_max.slice(0, 10)}`}),
  ]));

  const gates = card({title: "Data quality gates",
    sub: "Named, re-runnable checks. Several are <em>expected</em> to warn — that is exactly "
       + "what motivates the Phase 3 cleaning recipe, and each warning below maps to a "
       + "specific, auditable drop step."});
  el.appendChild(gates);
  gates.body.appendChild(table({raw: true,
    cols: [{label: "Gate"}, {label: "Status"}, {label: "Detail", wrap: true}],
    rows: prof.gates.map(g => [
      `<span class="mono">${esc(g.gate)}</span>`,
      g.status === "pass" ? `${dot("good")} pass` : `${dot("serious")} warn`,
      esc(g.detail)])}));

  // Cards are appended BEFORE their charts are drawn: a chart measures its
  // container, and a container that is still detached measures zero.
  const cHist = card({title: "Basket size distribution",
    sub: "Clipped at 60 items. The long right tail is wholesale ordering — a different "
       + "purchasing process, removed in Phase 3."});
  const cMonth = card({title: "Invoice volume by month",
    sub: "A pronounced Q4 peak. This is why the holdout is split on time rather than at "
       + "random — the last 30% of invoices spans the Christmas ramp."});
  el.appendChild(grid("g2", [cHist, cMonth]));
  C.bar(cHist.body, {
    data: prof.basket_size_hist.map(b => ({label: Math.round(b.bin_start), value: b.count})),
    xEvery: 4, yLabel: "invoices", xName: "Basket size (items)", yName: "Invoices",
    tipTitle: (d) => `${d.label} items`, tipRows: (d) => [["invoices", f.n(d.value)]]});
  C.bar(cMonth.body, {data: prof.monthly_invoices.map(m => ({label: m.month.slice(2), value: m.invoices})),
    yLabel: "invoices", xName: "Month", yName: "Invoices",
    tipTitle: (d) => d.label, tipRows: (d) => [["invoices", f.n(d.value)]]});

  const ti = card({title: "Most frequently sold products",
    sub: "Raw line-count popularity, before basket de-duplication."});
  el.appendChild(ti);
  C.hbar(ti.body, {data: prof.top_items.slice(0, 15).map(i => ({label: i.stock_code, value: i.count})),
    labelW: 90, xName: "Stock code", yName: "Line items"});

  const cc = card({title: "Column profile",
    sub: "Per-column types, missingness and distribution shape."});
  el.appendChild(cc);
  cc.body.appendChild(table({raw: true, tall: true,
    cols: [{label: "Column"}, {label: "Type"}, {label: "Missing", num: true},
           {label: "% missing", num: true}, {label: "Unique", num: true},
           {label: "Min", num: true}, {label: "Median", num: true}, {label: "Max", num: true},
           {label: "Negative", num: true}],
    rows: prof.columns.map(c => [
      `<span class="mono">${esc(c.column)}</span>`, `<span class="pill">${esc(c.dtype)}</span>`,
      f.n(c.n_missing),
      `${f.f(c.pct_missing, 2)}% ${inlineBar(c.pct_missing / 100, "var(--serious)")}`,
      f.n(c.n_unique),
      c.min !== undefined ? f.sci(c.min) : "–",
      c.p50 !== undefined ? f.sci(c.p50) : "–",
      c.max !== undefined ? f.sci(c.max) : "–",
      c.n_negative !== undefined ? f.n(c.n_negative) : "–"])}));

  const bc = card({title: "Where the invoices come from"});
  el.appendChild(bc);
  C.hbar(bc.body, {data: prof.by_country.map(c => ({label: c.country, value: c.invoices})),
    labelW: 150, xName: "Country", yName: "Invoices"});
  bc.body.appendChild(h(note(`The population is <b>overwhelmingly UK</b>. Mixing a dominant
    market with a long tail of small ones means the rules describe the UK; the config
    exposes a <span class="mono">countries</span> filter to mine a single homogeneous
    market instead.`)));
}

/* ------------------------------------------------------------------ */
export async function prep(el) {
  section(el, "");
  const p = await A.preparation();
  el.appendChild(h(`<div class="view-head">
    <p>CRISP-DM Phase 3. Every row removed is accounted for below with its
    justification — a reviewer can reconstruct the exact dataset from this log.</p></div>`));

  const total = p.drop_log[p.drop_log.length - 1];
  el.appendChild(grid("g4", [
    tile({label: "Rows kept", value: f.n(total.rows_after),
      foot: `of ${f.n(total.rows_before)} raw lines (<b>−${f.f(total.pct_dropped, 2)}%</b>)`}),
    tile({label: "Baskets built", value: f.n(p.dataset.n_transactions),
      foot: `avg <b>${f.f(p.dataset.avg_basket_size, 1)}</b> items · density
             <b>${f.f(p.dataset.density * 100, 2)}%</b>`}),
    tile({label: "Items retained", value: f.n(p.dataset.n_items),
      foot: `after the basket-frequency floor`}),
    tile({label: "Explore / holdout", value: `${f.compact(p.split.explore_n)} / ${f.compact(p.split.holdout_n)}`,
      foot: `<b>${esc(p.split.strategy)}</b> split at ${esc((p.split.explore_end || "").slice(0, 10))}`}),
  ]));

  const c1 = card({title: "Cleaning ledger",
    sub: "Applied in order. Each step is a single, named predicate with a rationale — no "
       + "unexplained <span class='mono'>dropna()</span>."});
  el.appendChild(c1);
  c1.body.appendChild(table({raw: true,
    cols: [{label: "Step"}, {label: "Before", num: true}, {label: "After", num: true},
           {label: "Dropped", num: true}, {label: "%", num: true}, {label: "Rationale", wrap: true}],
    rows: p.drop_log.map(d => [
      d.step === "TOTAL" ? `<b>${esc(d.step)}</b>` : `<span class="mono">${esc(d.step)}</span>`,
      f.n(d.rows_before), f.n(d.rows_after), f.n(d.rows_dropped),
      `${f.f(d.pct_dropped, 2)}% ${inlineBar(d.pct_dropped / 4, "var(--series-2)")}`,
      esc(d.rationale)])}));

  const c2 = card({title: "Row attrition",
    sub: "How many invoice lines survive each cleaning step."});
  el.appendChild(c2);
  C.bar(c2.body, {height: 200,
    data: p.drop_log.filter(d => d.step !== "TOTAL").map(d => ({
      label: d.step.replace(/^drop_/, "").replace(/_/g, " ").slice(0, 14), value: d.rows_after})),
    yLabel: "rows remaining", xName: "Step", yName: "Rows remaining",
    tipTitle: (d) => d.label, tipRows: (d) => [["rows remaining", f.n(d.value)]]});

  const c3 = card({title: "Basket construction",
    sub: "Line items → sets. Two further filters apply after cleaning."});
  el.appendChild(c3);
  c3.body.appendChild(table({raw: true,
    cols: [{label: "Step"}, {label: "Effect", wrap: true}, {label: "Rationale", wrap: true}],
    rows: p.prep_log.map(s => [`<span class="mono">${esc(s.step)}</span>`,
      esc(s.detail), esc(s.rationale)])}));
  c3.body.appendChild(h(note(`<b>Sets, not counts.</b> A basket containing six of one item is
    a single piece of co-occurrence evidence, not six. De-duplicating line items before
    counting is what makes support a probability rather than a volume
    (Agrawal &amp; Srikant, VLDB 1994).`)));

  const c4 = card({title: "Temporal holdout",
    sub: "The split that makes every downstream validation claim meaningful."});
  el.appendChild(c4);
  c4.body.appendChild(kv([
    ["Strategy", `<b>${esc(p.split.strategy)}</b> — holdout fraction ${p.split.holdout_fraction}`],
    ["Explore window", `${esc((p.split.explore_start || "").slice(0, 10))} → ${esc((p.split.explore_end || "").slice(0, 10))} &nbsp; (${f.n(p.split.explore_n)} baskets)`],
    ["Holdout window", `${esc((p.split.holdout_start || "").slice(0, 10))} → ${esc((p.split.holdout_end || "").slice(0, 10))} &nbsp; (${f.n(p.split.holdout_n)} baskets)`],
  ]));
  c4.body.appendChild(h(note(`<b>Why time and not a random split.</b> Webb (2007) shows that
    testing a rule on the data that suggested it inflates the false-discovery rate without
    bound, so the holdout must be untouched during search. For retail we go further: a
    random split would let one December basket vouch for a rule discovered in another
    December basket. Splitting on time makes the holdout a genuine forecast — and it is
    the <em>harder</em> test, because it spans the Christmas peak the training window
    never sees.`)));

  const c5 = card({title: "Top items after preparation",
    sub: "Basket-presence support, which is what the miner actually consumes."});
  el.appendChild(c5);
  C.hbar(c5.body, {data: p.top_items.slice(0, 18).map(i => ({label: i.label, value: i.support})),
    labelW: 235, xfmt: (v) => f.pct(v, 1), xName: "Product", yName: "Basket support",
    tipRows: (d) => [["support", f.pct(d.value, 2)]]});
}
