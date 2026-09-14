/* Views for CRISP-DM phases 4-6: modeling, AutoResearch, evaluation, deployment. */
import * as C from "./charts.js";
import {A} from "./api.js";
import {h, esc, tile, card, table, kv, dot, chip, note, cite, grid, section, inlineBar} from "./ui.js";
import {ruleHtml} from "./views_data.js";

const f = C.fmt;

/* ================= frequent itemsets ================= */
export async function itemsets(el) {
  section(el, "");
  const it = await A.itemsets();
  const st = it.algorithm_stats;
  el.appendChild(h(`<div class="view-head">
    <p>CRISP-DM Phase 4a. FP-Growth (Han, Pei &amp; Yin, SIGMOD 2000) implemented from
    scratch: it compresses the transaction database into a prefix tree ordered by
    descending item frequency, then mines it recursively through conditional trees —
    never generating candidate itemsets at all, which is what makes it beat Apriori
    by an order of magnitude here.</p></div>`));

  el.appendChild(grid("g4", [
    tile({label: "Frequent itemsets", value: f.n(it.n_itemsets),
      foot: `at min support <b>${f.pct(it.min_support, 2)}</b> (count ≥ ${f.n(it.min_count)})`}),
    tile({label: "Mining time", value: f.f(st.seconds, 3), unit: "s",
      foot: `<b>${f.n(st.nodes_created)}</b> FP-tree nodes built`}),
    tile({label: "Conditional trees", value: f.n(st.conditional_trees),
      foot: `max recursion depth <b>${st.max_depth}</b>`}),
    tile({label: "Candidates touched", value: f.compact(st.candidates_generated),
      foot: `Apriori would enumerate combinatorially more`}),
  ]));

  const c1 = card({title: "Itemsets by size",
    sub: "The lattice thins fast — the combinatorial explosion is controlled by the "
       + "downward-closure property: no superset of an infrequent set can be frequent."});
  el.appendChild(c1);
  C.bar(c1.body, {height: 200, direct: true,
    data: Object.entries(it.per_level).sort((a, b) => a[0] - b[0])
      .map(([k, v]) => ({label: `${k}-itemset`, value: v})),
    yLabel: "itemsets", xName: "Itemset size", yName: "Count",
    tipTitle: (d) => d.label, tipRows: (d) => [["itemsets", f.n(d.value)]]});

  const c2 = card({title: "Largest frequent itemsets",
    sub: "Multi-item sets ranked by support. These are the raw material every rule is cut from.",
    right: chip(`${it.top.length} shown`)});
  el.appendChild(c2);
  c2.body.appendChild(table({raw: true, tall: true,
    cols: [{label: "Size", num: true}, {label: "Itemset", wrap: true},
           {label: "Count", num: true}, {label: "Support", num: true}, {label: ""}],
    rows: it.top.slice(0, 200).map(s => [
      s.size, esc(s.items.join("  ·  ")), f.n(s.count), f.pct(s.support, 2),
      inlineBar(s.support / it.top[0].support)])}));
}

/* ================= rule explorer ================= */
export async function rules(el) {
  section(el, "");
  el.appendChild(h(`<div class="view-head">
    <p>CRISP-DM Phase 4b. Every rule carries all 32 interestingness measures, an exact
    Fisher p-value, a search-space-corrected q-value, Webb's productivity test, and its
    behaviour on the temporal holdout. Click a row for the full contingency table.</p></div>`));

  const bar = h(`<div class="filters">
    <div class="field"><label>Search</label><input id="r-q" type="search" placeholder="product name…" style="width:210px"></div>
    <div class="field"><label>Sort by</label><select id="r-sort">
      <option value="leverage">Leverage</option><option value="lift">Lift</option>
      <option value="confidence">Confidence</option><option value="support">Support</option>
      <option value="kulczynski">Kulczynski</option><option value="zhang">Zhang's metric</option>
      <option value="conviction">Conviction</option><option value="phi">Phi coefficient</option>
      <option value="j_measure">J-measure</option><option value="chi_square">Chi-square</option>
      <option value="holdout_lift">Holdout lift</option>
    </select></div>
    <div class="field"><label>Min lift <span id="r-lift-v" class="pill">1.0</span></label>
      <input id="r-lift" type="range" min="1" max="12" step="0.25" value="1"></div>
    <div class="field"><label>Min confidence <span id="r-conf-v" class="pill">0%</span></label>
      <input id="r-conf" type="range" min="0" max="0.95" step="0.05" value="0"></div>
    <div class="field"><label>Antecedent size</label><select id="r-alen">
      <option value="0">any</option><option value="1">1 item</option>
      <option value="2">2 items</option><option value="3">3 items</option></select></div>
    <div class="field"><label>Filters</label><div style="display:flex;gap:6px">
      <button class="btn" id="r-val">Holdout-validated</button>
      <button class="btn" id="r-sig">FDR-significant</button>
      <button class="btn" id="r-prod">Productive</button>
    </div></div>
  </div>`);
  el.appendChild(bar);

  const head = card({title: "Rules", sub: ""});
  el.appendChild(head);
  const detail = card({title: "Rule detail", sub: "Select a rule above."});
  el.appendChild(detail);

  const state = {sort: "leverage", min_lift: 1, min_confidence: 0, antecedent_len: 0,
    search: "", validated_only: false, significant_only: false, productive_only: false};

  const on = (id, key) => {
    const b = bar.querySelector(id);
    b.onclick = () => { state[key] = !state[key]; b.classList.toggle("on", state[key]); load(); };
  };
  on("#r-val", "validated_only"); on("#r-sig", "significant_only"); on("#r-prod", "productive_only");
  bar.querySelector("#r-sort").onchange = (e) => { state.sort = e.target.value; load(); };
  bar.querySelector("#r-alen").onchange = (e) => { state.antecedent_len = +e.target.value; load(); };
  bar.querySelector("#r-q").oninput = debounce((e) => { state.search = e.target.value; load(); }, 220);
  bar.querySelector("#r-lift").oninput = debounce((e) => {
    state.min_lift = +e.target.value; bar.querySelector("#r-lift-v").textContent = f.f(state.min_lift, 2); load(); }, 160);
  bar.querySelector("#r-conf").oninput = debounce((e) => {
    state.min_confidence = +e.target.value; bar.querySelector("#r-conf-v").textContent = f.pct(state.min_confidence, 0); load(); }, 160);

  async function load() {
    head.body.innerHTML = `<div class="loading">loading…</div>`;
    const d = await A.rules({...state, limit: 300, desc: true});
    const subEl = head.querySelector(".sub");
    subEl.hidden = false;
    subEl.innerHTML =
      `<b>${f.n(d.matched)}</b> of ${f.n(d.total)} rules match · mined at support
       ≥ <b>${f.pct(d.params.min_support, 2)}</b>, confidence ≥ <b>${f.pct(d.params.min_confidence, 0)}</b>,
       lift ≥ <b>${f.f(d.params.min_lift, 2)}</b>, antecedent ≤ <b>${d.params.max_antecedent_len}</b>`;
    head.body.innerHTML = "";
    if (!d.rules.length) { head.body.innerHTML = `<div class="loading">No rules match these filters.</div>`; return; }
    const t = table({raw: true, tall: true,
      cols: [{label: "Rule", wrap: true}, {label: "Supp", num: true}, {label: "Conf", num: true},
             {label: "Lift", num: true}, {label: "Lev", num: true}, {label: "Kulc", num: true},
             {label: "Zhang", num: true}, {label: "q (FDR)", num: true},
             {label: "Prod"}, {label: "Holdout conf", num: true}, {label: "Status"}],
      rows: d.rules.map(r => [
        ruleHtml(r), f.pct(r.support, 2), f.pct(r.confidence, 1), f.f(r.lift, 2),
        f.f(r.leverage, 4), f.f(r.kulczynski, 3), f.f(r.zhang, 3),
        f.p(r.q_value_searchspace),
        r.productive ? `${dot("good")}` : `${dot("warn")}`,
        r.holdout_confidence != null ? f.pct(r.holdout_confidence, 1) : "–",
        r.validated ? `${dot("good")} validated`
          : r.holdout_count < 5 ? `${dot("warn")} sparse` : `${dot("serious")} drifted`])});
    head.body.appendChild(t);
    t.querySelectorAll("tbody tr").forEach((tr, i) => {
      tr.style.cursor = "pointer";
      tr.onclick = () => showDetail(d.rules[i]);
    });
    await showDetail(d.rules[0]);
  }

  async function showDetail(r) {
    const props = await A.properties();
    const gloss = Object.fromEntries(props.glossary.map(g => [g.key, g]));
    detail.querySelector("h3").textContent = "Rule detail";
    const dsub = detail.querySelector(".sub"); dsub.hidden = false;
    dsub.innerHTML = ruleHtml(r) +
      ` &nbsp;<span class="pill">${esc(r.rule_id)}</span>`;
    detail.body.innerHTML = "";
    const n11 = r.n11, n1_ = r.n1_, n_1 = r.n_1, N = r.N;
    const n10 = n1_ - n11, n01 = n_1 - n11, n00 = N - n1_ - n_1 + n11;
    const g = h(`<div class="grid g2"></div>`);
    const left = h(`<div></div>`);
    left.appendChild(h(`<h4 style="font-size:12px;margin-bottom:8px">2×2 contingency table</h4>`));
    left.appendChild(table({raw: true,
      cols: [{label: ""}, {label: "consequent", num: true}, {label: "¬ consequent", num: true}, {label: "total", num: true}],
      rows: [["<b>antecedent</b>", `<b>${f.n(n11)}</b>`, f.n(n10), f.n(n1_)],
             ["<b>¬ antecedent</b>", f.n(n01), f.n(n00), f.n(N - n1_)],
             ["<b>total</b>", f.n(n_1), f.n(N - n_1), `<b>${f.n(N)}</b>`]]}));
    left.appendChild(h(`<div style="margin-top:14px"></div>`));
    left.appendChild(kv([
      ["Fisher exact p", `<b>${f.p(r.p_fisher)}</b> <span class="muted">(one-sided, hypergeometric)</span>`],
      ["q-value (reported set)", f.p(r.q_value_reported)],
      ["q-value (search space)", `<b>${f.p(r.q_value_searchspace)}</b>`],
      ["Productive (Webb 2007)", r.productive ? `${dot("good")} yes — beats every generalisation`
        : `${dot("serious")} no — a sub-rule explains it`],
      ["Holdout support", r.holdout_support != null ? f.pct(r.holdout_support, 2) : "–"],
      ["Holdout confidence", r.holdout_confidence != null
        ? `${f.pct(r.holdout_confidence, 1)} <span class="muted">(in-sample ${f.pct(r.confidence, 1)})</span>` : "–"],
      ["Holdout lift", r.holdout_lift != null
        ? `${f.f(r.holdout_lift, 2)} <span class="muted">(in-sample ${f.f(r.lift, 2)})</span>` : "–"],
      ["Replicated", r.validated ? `${dot("good")} yes` : `${dot("serious")} no`],
    ]));
    g.appendChild(left);

    const right = h(`<div></div>`);
    right.appendChild(h(`<h4 style="font-size:12px;margin-bottom:8px">All interestingness measures</h4>`));
    const keys = props.glossary.map(x => x.key).filter(k => r[k] !== undefined && r[k] !== null);
    right.appendChild(table({raw: true, tall: true,
      cols: [{label: "Measure"}, {label: "Value", num: true}, {label: "Family"}, {label: "Source", wrap: true}],
      rows: keys.map(k => [
        `<span title="${esc(gloss[k].formula)}">${esc(gloss[k].label)}</span>`,
        f.sci(r[k]), `<span class="pill">${esc(gloss[k].family)}</span>`,
        `<span class="cite">${esc(gloss[k].citation)}</span>`])}));
    g.appendChild(right);
    detail.body.appendChild(g);
  }
  await load();
}

function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

/* ================= AutoResearch ================= */
export async function autoresearch(el) {
  section(el, "");
  const ar = await A.autoresearch();
  if (!ar.enabled) {
    el.appendChild(h(`<div class="err">AutoResearch was disabled for this run. Re-run with
      <span class="mono">armlab run</span> (without <span class="mono">--no-autoresearch</span>).</div>`));
    return;
  }
  const cols = C.SERIES();
  el.appendChild(h(`<div class="view-head">
    <p>An automated researcher: it proposes a mining configuration, mines, validates on the
    untouched temporal holdout, scores the result against a literature-grounded objective,
    and climbs. ${f.n(ar.n_evaluations)} configurations were evaluated across
    ${ar.n_restarts_used + 1} search runs; every one is logged below.</p></div>`));

  el.appendChild(grid("g4", [
    tile({label: "Champion objective", value: f.f(ar.best_score, 4),
      foot: `baseline <b>${f.f(ar.seed_score, 4)}</b> → <b>+${f.pct(ar.improvement_pct / 100, 1)}</b>`}),
    tile({label: "Configurations tried", value: f.n(ar.n_evaluations),
      foot: `${f.n(ar.cache.hits)} itemset-cache hits saved re-mining`}),
    tile({label: "Search wall clock", value: f.f(ar.seconds, 1), unit: "s",
      foot: `${f.f(ar.seconds / ar.n_evaluations, 2)}s per configuration`}),
    tile({label: "Restarts used", value: ar.n_restarts_used,
      foot: `escape hatch for local optima`}),
  ]));

  /* convergence */
  const c1 = card({title: "Search trajectory",
    sub: "Each dot is one mining configuration evaluated end to end. The line is the best "
       + "objective found so far — the staircase is the hill climber committing to a move."});
  el.appendChild(c1);
  const conv = ar.convergence;
  const accepted = new Set(ar.trials.filter(t => t.accepted).map(t => t.index));
  C.line(c1.body, {height: 290, xIsIndex: true,
    series: [
      {name: "Evaluated configuration", color: cols[1], type: "dots", r: 3.4,
       points: conv.map((c, i) => ({x: c.evaluation, y: c.score, dim: !accepted.has(i),
         label: `Evaluation ${c.evaluation}`}))},
      {name: "Accepted move", color: cols[2], type: "dots", r: 5,
       points: conv.filter((c, i) => accepted.has(i)).map(c => ({x: c.evaluation, y: c.score,
         label: `Accepted at eval ${c.evaluation}`}))},
      {name: "Best so far", color: cols[0], width: 2, points: conv.map(c => ({x: c.evaluation, y: c.best}))},
    ],
    xLabel: "evaluation", yLabel: "objective", yfmt: (v) => f.f(v, 3), xfmt: (v) => f.n(v),
    markers: false,
    tipRows: (p) => { const t = ar.trials[p.x - 1] || {}; return [
      ["objective", f.f(p.y, 4)], ["min support", f.pct(t.params?.min_support ?? 0, 2)],
      ["min confidence", f.pct(t.params?.min_confidence ?? 0, 0)],
      ["min lift", f.f(t.params?.min_lift ?? 0, 2)],
      ["validated rules", f.n(t.diagnostics?.n_validated ?? 0)],
      ["restart", t.restart], ["step size", f.f(t.step_size ?? 0, 3)]];}});
  c1.body.appendChild(h(note(`<b>Why hill climbing and not Bayesian optimisation?</b> The
    landscape is riddled with plateaus — nudging min-support by 0.001 usually changes
    nothing, then crosses a support threshold and changes everything. A Gaussian-process
    surrogate fits that badly, and at ~60 evaluations on 5 dimensions it would spend most
    of its budget learning the surrogate. Stochastic local search with random restarts is
    the standard answer for this shape of landscape (Hoos &amp; Stützle 2004; Selman et al., AAAI 1992).`)));

  /* search space scatter */
  const c2 = card({title: "Where the search looked",
    sub: "Every evaluated configuration in the two dimensions that matter most. Support is "
       + "log-scaled because the itemset count collapses super-linearly as it falls."});
  el.appendChild(c2);
  C.scatter(c2.body, {height: 320, logX: true,
    points: ar.trials.map(t => ({x: t.params.min_support, y: t.params.min_confidence,
      c: t.score, r: t.is_best_so_far ? 7 : 5, label: `Evaluation ${t.index + 1}`})),
    xLabel: "min support (log)", yLabel: "min confidence", cLabel: "objective",
    xfmt: (v) => f.pct(v, 2), yfmt: (v) => f.pct(v, 0),
    tipRows: (p) => { const t = ar.trials[p.label.split(" ")[1] - 1]; return [
      ["objective", f.f(t.score, 4)], ["min support", f.pct(t.params.min_support, 3)],
      ["min confidence", f.pct(t.params.min_confidence, 1)],
      ["min lift", f.f(t.params.min_lift, 2)],
      ["max antecedent", t.params.max_antecedent_len],
      ["rules", f.n(t.diagnostics?.n_rules ?? 0)],
      ["validated", f.n(t.diagnostics?.n_validated ?? 0)]];}});

  /* objective definition */
  const c3 = card({title: "The objective function",
    sub: "Association mining has no natural loss, so 'tune min-support until the output looks "
       + "nice' is the norm. This makes the target explicit, multi-objective and cited "
       + "(after Ghosh &amp; Nath, <i>Information Sciences</i> 163, 2004). Every term is "
       + "normalised to [0,1], higher is better."});
  el.appendChild(c3);
  const W = ar.objective.weights;
  c3.body.appendChild(table({raw: true,
    cols: [{label: "Term"}, {label: "Weight", num: true}, {label: "Definition", wrap: true},
           {label: "Why it is in the objective", wrap: true}, {label: "Source", wrap: true}],
    rows: Object.entries(ar.objective.terms).map(([k, d]) => [
      `<b>${esc(d.label)}</b>`,
      `${f.f(W[k], 2)} ${inlineBar(W[k])}`,
      esc(d.definition), esc(d.why), `<span class="cite">${esc(d.citation)}</span>`])}));

  const c4 = card({title: "Objective decomposition, baseline vs champion",
    sub: "Where the improvement actually came from. Direct labels give the totals; the "
       + "table view has every value."});
  el.appendChild(c4);
  const seed = ar.trials.find(t => t.kind === "seed") || ar.trials[0];
  const best = ar.trials.reduce((a, b) => b.score > a.score ? b : a, ar.trials[0]);
  const tkeys = Object.keys(best.weighted);
  C.stacked(c4.body, {categories: ["Baseline (seed)", "Champion"],
    series: tkeys.map((k, i) => ({name: k.replace(/_/g, " "), color: cols[i],
      values: [seed.weighted[k], best.weighted[k]]})),
    yfmt: (v) => f.f(v, 3), yLabel: "weighted contribution",
    tipRows: (s, cat, v) => [["term", s.name], ["weighted", f.f(v, 4)]]});

  /* search space */
  const c5 = card({title: "Search space",
    sub: "Each dimension is mapped to a normalised [0,1] coordinate so a step of 0.1 means a "
       + "comparable move in every dimension — hill climbing needs a metric space, and raw "
       + "hyperparameters are not one."});
  el.appendChild(c5);
  c5.body.appendChild(table({raw: true,
    cols: [{label: "Dimension"}, {label: "Range", num: true}, {label: "Scale"},
           {label: "Champion", num: true}, {label: "Why this scale / range", wrap: true}],
    rows: ar.space.map(d => [
      `<span class="mono">${esc(d.name)}</span>`,
      `${f.sci(d.low)} – ${f.sci(d.high)}`, `<span class="pill">${esc(d.scale)}</span>`,
      `<b>${f.sci(ar.best_params[d.name])}</b>`, esc(d.rationale)])}));

  /* Did the search actually explore rule complexity, or collapse early? */
  const byShape = new Map();
  for (const t of ar.trials) {
    const k = `${t.params.max_itemset_len}|${t.params.max_antecedent_len}`;
    if (!byShape.has(k)) byShape.set(k, []);
    byShape.get(k).push(t);
  }
  const shapeRows = [...byShape.entries()].map(([k, ts]) => {
    const [il, al] = k.split("|").map(Number);
    const best = ts.reduce((a, b) => b.score > a.score ? b : a);
    return {il, al, n: ts.length, best,
      mean: ts.reduce((a, b) => a + b.score, 0) / ts.length};
  }).sort((a, b) => a.il - b.il || a.al - b.al);
  const champShape = shapeRows.reduce((a, b) => b.best.score > a.best.score ? b : a);
  const mostValidated = shapeRows.reduce((a, b) =>
    (b.best.diagnostics?.n_validated ?? 0) > (a.best.diagnostics?.n_validated ?? 0) ? b : a);

  const cX = card({title: "Did the search explore rule complexity?",
    sub: "Objective grouped by the two structural dimensions. This is the panel that answers "
       + "the obvious objection — that the optimiser simply never tried longer rules."});
  el.appendChild(cX);
  cX.body.appendChild(table({raw: true,
    cols: [{label: "Max itemset"}, {label: "Max antecedent"}, {label: "Trials", num: true},
           {label: "Best objective", num: true}, {label: "Mean objective", num: true},
           {label: "Rules at best", num: true}, {label: "Validated at best", num: true}],
    rows: shapeRows.map(r => [
      `${r.il}${r.il === champShape.il && r.al === champShape.al ? ' <span class="chip good">champion</span>' : ""}`,
      String(r.al), f.n(r.n),
      `<b>${f.f(r.best.score, 4)}</b> ${inlineBar(r.best.score)}`, f.f(r.mean, 4),
      f.n(r.best.diagnostics?.n_rules ?? 0),
      `${f.n(r.best.diagnostics?.n_validated ?? 0)}${
        r.il === mostValidated.il && r.al === mostValidated.al ? ' <span class="chip">most</span>' : ""}`])}));
  cX.body.appendChild(h(note(`<b>The interesting result.</b> The champion mines
    <b>pairwise rules only</b> (itemsets of size ${champShape.il}, single-item antecedents) —
    yet the <span class="mono">${mostValidated.il}/${mostValidated.al}</span> configuration
    found <b>${f.n(mostValidated.best.diagnostics?.n_validated ?? 0)}</b> holdout-validated
    rules against the champion's
    <b>${f.n(champShape.best.diagnostics?.n_validated ?? 0)}</b>. It still lost, because the
    parsimony and compute-economy terms charge for longer antecedents and slower mines, and
    that cost exceeds the marginal yield. Whether that is the right trade is a
    <em>business</em> decision encoded in the weights above, not a fact about the data —
    raise <span class="mono">w_validated_yield</span> and the champion changes shape.
    Longer rules were explored across ${f.n(ar.trials.length)} trials; they were not
    excluded by construction.`)));

  const c6 = card({title: "Optimiser configuration"});
  el.appendChild(c6);
  c6.body.appendChild(kv(Object.entries(ar.config).map(([k, v]) =>
    [k.replace(/_/g, " "), `<b>${esc(v)}</b>`])));

  const c7 = card({title: "Experiment ledger",
    sub: "Append-only. Every configuration is written to disk before the next one starts, so a "
       + "crashed search is still analysable.",
    right: chip(`${ar.trials.length} trials`)});
  el.appendChild(c7);
  c7.body.appendChild(table({raw: true, tall: true,
    cols: [{label: "#", num: true}, {label: "Kind"}, {label: "Restart", num: true},
           {label: "Support", num: true}, {label: "Conf", num: true}, {label: "Lift", num: true},
           {label: "Ante", num: true}, {label: "Rules", num: true}, {label: "Validated", num: true},
           {label: "Coverage", num: true}, {label: "Objective", num: true}, {label: "Move"}],
    rows: ar.trials.map(t => [
      t.index + 1, `<span class="pill">${esc(t.kind)}</span>`, t.restart,
      f.pct(t.params.min_support, 2), f.pct(t.params.min_confidence, 0),
      f.f(t.params.min_lift, 2), t.params.max_antecedent_len,
      f.n(t.diagnostics?.n_rules ?? 0), f.n(t.diagnostics?.n_validated ?? 0),
      f.pct(t.diagnostics?.coverage ?? 0, 1),
      `<b>${f.f(t.score, 4)}</b> ${inlineBar(t.score)}`,
      t.is_best_so_far ? `${dot("good")} new best` : t.accepted ? `${dot("good")} accepted` : `<span class="muted">rejected</span>`])}));
}

/* ================= network ================= */
export async function network(el) {
  section(el, "");
  const n = await A.network();
  el.appendChild(h(`<div class="view-head">
    <p>The top rules as an item graph. Node size is how central a product is to the rule set;
    edge shade is lift. Clusters are product families that co-purchase — the structure a
    merchandiser would recognise as a "collection".</p></div>`));
  const c = card({title: "Co-purchase network",
    sub: `${n.nodes.length} items connected by ${n.edges.length} rule edges, from the top `
       + `${n.n_rules_shown} rules by leverage.`});
  el.appendChild(c);
  C.network(c.body, {nodes: n.nodes, edges: n.edges, height: 560});
}

/* ================= evaluation ================= */
export async function evaluation(el) {
  section(el, "");
  const ev = await A.evaluation();
  const cols = C.SERIES();
  const ho = ev.holdout, sg = ev.significance || {}, nb = ev.null_baseline || {};
  el.appendChild(h(`<div class="view-head">
    <p>CRISP-DM Phase 5. Association rules have no accuracy score, which is why so many
    market-basket projects ship rules that do not replicate. This page treats the rule set
    as a predictive artefact and scores it on data the miner never saw.</p></div>`));

  el.appendChild(grid("g4", [
    tile({label: "Replication rate", value: f.pct(ho.replication_rate),
      foot: `<b>${f.n(ho.n_replicated)}</b> of ${f.n(ho.n_rules)} rules clear their thresholds out of sample`}),
    tile({label: "Lift rank stability", value: f.f(ho.lift_rank_spearman, 3),
      foot: `Spearman ρ, in-sample vs holdout lift`}),
    tile({label: "Firing precision", value: f.pct(ho.firing_precision),
      foot: `when a rule fires on a holdout basket, how often the consequent is there`}),
    tile({label: "Confidence drift", value: f.f(ho.confidence_drift_mean, 4),
      foot: `MAE <b>${f.f(ho.confidence_drift_mae, 4)}</b> — selection optimism`}),
  ]));

  const c0 = card({title: "Statistical validity",
    sub: "Mining is a massive multiple-comparisons exercise. Reporting rules without this "
       + "section is the single most common flaw in market-basket write-ups."});
  el.appendChild(c0);
  c0.body.appendChild(table({raw: true,
    cols: [{label: "Quantity", wrap: true}, {label: "Value", num: true}, {label: "Meaning", wrap: true}],
    rows: [
      ["Hypotheses in the search space", f.compact(sg.n_hypotheses_corrected_for ?? 0),
       "Every antecedent/consequent split the miner could have examined at these length "
       + "limits. Webb (2007) argues the correction must count this, not the rules you chose to report."],
      ["Rules tested", f.n(sg.n_rules_tested ?? 0),
       "Rules that cleared the support/confidence/lift thresholds and received a Fisher exact test."],
      ["Significant, uncorrected", f.n(sg.n_significant_uncorrected ?? 0),
       `At α = ${sg.alpha ?? 0.05} with no correction — the naive number, shown so the gap below is visible.`],
      ["Expected false positives, uncorrected", f.n(sg.expected_false_positives_uncorrected ?? 0),
       "α × number of tests. This is the number the correction exists to remove."],
      ["Significant after correction", `<b>${f.n(sg.n_significant_corrected ?? 0)}</b>`,
       `Benjamini–Hochberg FDR against the full search space.`],
      ["Productive (Webb 2007)", f.n(sg.n_productive ?? 0),
       "Rules that beat every one of their own generalisations — kills the artefact where "
       + "{a,b}⇒c is reported only because a⇒c is strong."],
      ["Significant AND productive", `<b>${f.n(sg.n_significant_and_productive ?? 0)}</b>`,
       "The set a reviewer should actually look at."],
    ]}));
  const ante = ev.champion?.params?.max_antecedent_len ?? 0;
  if (ante <= 1) {
    c0.body.appendChild(h(note(`<b>Read the productivity row with care.</b> The champion
      configuration mines <b>single-item antecedents only</b>, and Webb's productivity test
      compares a rule against its own <em>proper</em> generalisations — of which a one-item
      antecedent has none. The test is therefore <b>vacuously true for every rule here</b>,
      not evidence that every rule passed a demanding filter. It becomes a real filter the
      moment <span class="mono">max_antecedent_len ≥ 2</span>, which is exactly the setting
      the AutoResearch panel shows being explored and rejected on other grounds.`, "warn")));
  }
  if ((sg.n_significant_corrected ?? 0) === (sg.n_rules_tested ?? -1)) {
    c0.body.appendChild(h(note(`<b>Why every rule survives the correction.</b> These are not
      marginal effects: the median Fisher p-value is far below any threshold the
      ${f.compact(sg.n_hypotheses_corrected_for ?? 0)}-hypothesis correction imposes, and
      many underflow to exactly 0 in double precision. That is a consequence of the support
      floor — a rule needs hundreds of co-occurrences to clear it, and at those counts the
      hypergeometric tail is vanishing. The correction is doing real work at
      <em>lower</em> support thresholds, which the sensitivity sweep below reaches.`)));
  }

  const c1 = card({title: "Swap-randomised null baseline",
    sub: "A distribution-free check that sits alongside the analytic FDR."});
  el.appendChild(c1);
  c1.body.appendChild(kv([
    ["Method", esc(nb.method || "–")],
    ["Swaps performed", `<b>${f.n(nb.n_swaps_performed ?? 0)}</b>`],
    ["Rules on real data", `<b>${f.n(nb.n_rules_real ?? 0)}</b>`],
    ["Rules on randomised data", `<b>${f.n(nb.n_rules_null ?? 0)}</b>`],
    ["Empirical FDR", `<b>${f.f(nb.empirical_fdr ?? 0, 4)}</b>`],
  ]));
  c1.body.appendChild(h(note(esc(nb.interpretation || ""))));

  const c2 = card({title: "Lift calibration",
    sub: "Does a rule that promises lift X deliver lift X out of sample? Points on the "
       + "diagonal are calibrated; points below it are optimistic."});
  el.appendChild(c2);
  if ((ev.lift_calibration || []).length) {
    const lc = ev.lift_calibration;
    const mx = Math.max(...lc.map(b => Math.max(b.explore_lift_mean, b.holdout_lift_mean)));
    C.line(c2.body, {height: 280,
      series: [
        {name: "Perfect calibration", color: cols[2], width: 2, dash: "5 4", opacity: .8,
         points: [{x: 0, y: 0}, {x: mx, y: mx}]},
        {name: "Observed", color: cols[0], type: "dots", r: 6,
         points: lc.map(b => ({x: b.explore_lift_mean, y: b.holdout_lift_mean,
           label: `${f.n(b.n)} rules`}))},
      ],
      xLabel: "in-sample lift", yLabel: "holdout lift", markers: false,
      xfmt: (v) => f.f(v, 1), yfmt: (v) => f.f(v, 1),
      tipRows: (p) => [["in-sample lift", f.f(p.x, 2)], ["holdout lift", f.f(p.y, 2)],
        ["rules in bin", p.label || "–"]]});
  }

  const c3 = card({title: "Sensitivity around the champion",
    sub: "One-at-a-time sweeps holding the other hyperparameters at their champion values. "
       + "This is what tells a reviewer whether the champion sits on a robust plateau or a "
       + "narrow spike that the search happened to land on."});
  el.appendChild(c3);
  const sens = ev.sensitivity || {};
  const sg2 = h(`<div class="grid g3"></div>`);
  c3.body.appendChild(sg2);          // attach before drawing, so widths are real
  const holders = [];
  for (const dim of Object.keys(sens)) {
    const sub = h(`<div></div>`);
    sub.appendChild(h(`<h4 style="font-size:12px;margin-bottom:6px">${esc(dim.replace(/_/g, " "))}</h4>`));
    const holder = h(`<div></div>`); sub.appendChild(holder); sg2.appendChild(sub);
    holders.push([dim, holder]);
  }
  for (const [dim, holder] of holders) {
    const pts = sens[dim];
    C.line(holder, {height: 210,
      series: [{name: "objective", color: cols[0], width: 2, directLabel: false,
        points: pts.map(p => ({x: p.value, y: p.objective,
          label: `${dim} = ${f.sci(p.value)}`}))}],
      xLabel: dim, yLabel: "objective", xfmt: (v) => f.sci(v), yfmt: (v) => f.f(v, 3),
      tipRows: (p) => { const q = pts.find(z => z.value === p.x); return [
        ["objective", f.f(p.y, 4)], ["rules", f.n(q.n_rules)],
        ["validated", f.n(q.n_validated)], ["coverage", f.pct(q.coverage, 1)],
        ["seconds", f.f(q.seconds, 2)]];}});
  }

  const c4 = card({title: "Algorithm cross-validation",
    sub: "FP-Growth, Apriori and Eclat, all implemented from scratch, on identical input at "
       + "an identical threshold. They are exact algorithms, so identical output is a "
       + "correctness requirement — divergence would mean a bug."});
  el.appendChild(c4);
  c4.body.appendChild(table({raw: true,
    cols: [{label: "Algorithm"}, {label: "Seconds", num: true}, {label: "Itemsets", num: true},
           {label: "Candidates touched", num: true}, {label: "DB passes", num: true},
           {label: "Tree nodes", num: true}, {label: "Agrees"}],
    rows: ev.benchmark.runs.map(r => [
      `<b>${esc(r.algorithm)}</b>${r.algorithm === ev.benchmark.fastest ? ' <span class="chip good">fastest</span>' : ""}`,
      f.f(r.seconds, 3), f.n(r.itemsets_found), f.compact(r.candidates_generated),
      r.passes || "–", f.n(r.nodes_created) || "–",
      r.agrees_with_fpgrowth ? `${dot("good")} identical` : `${dot("bad")} DIVERGED`])}));
  c4.body.appendChild(h(note(esc(ev.benchmark.note))));

  const c5 = card({title: "Baseline vs champion", sub: "Full objective decomposition."});
  el.appendChild(c5);
  c5.body.appendChild(table({raw: true,
    cols: [{label: "Term"}, {label: "Baseline", num: true}, {label: "Champion", num: true},
           {label: "Δ", num: true}],
    rows: [...Object.keys(ev.champion.objective.terms).map(k => [
      k.replace(/_/g, " "), f.f(ev.baseline.objective.terms[k], 4),
      f.f(ev.champion.objective.terms[k], 4),
      fmtDelta(ev.champion.objective.terms[k] - ev.baseline.objective.terms[k])]),
      ["<b>weighted total</b>", `<b>${f.f(ev.baseline.objective.value, 4)}</b>`,
       `<b>${f.f(ev.champion.objective.value, 4)}</b>`,
       fmtDelta(ev.champion.objective.value - ev.baseline.objective.value)]]}));
}

const fmtDelta = (d) => `<b style="color:${d > 0 ? "var(--good)" : d < 0 ? "var(--critical)" : "var(--text-muted)"}">${
  d > 0 ? "+" : ""}${f.f(d, 4)}</b>`;

/* ================= measures & properties ================= */
export async function measures(el) {
  section(el, "");
  const p = await A.properties();
  el.appendChild(h(`<div class="view-head">
    <p>The 32 interestingness measures this project computes, and which formal properties
    each one satisfies. Crucially, the matrix below is <b>derived numerically</b> — each cell
    is a randomised experiment over 2×2 contingency tables, not a value transcribed from
    the paper. Re-running <span class="mono">armlab properties</span> reproduces it exactly.</p></div>`));

  const c0 = card({title: "The properties",
    sub: "Piatetsky-Shapiro's three axioms, plus the invariance properties of Tan, Kumar &amp; "
       + "Srivastava (KDD 2002)."});
  el.appendChild(c0);
  c0.body.appendChild(table({raw: true,
    cols: [{label: "Key"}, {label: "Property", wrap: true}, {label: "Source", wrap: true}],
    rows: p.properties.map(x => [`<b class="mono">${esc(x.key)}</b>`, esc(x.description),
      `<span class="cite">${esc(x.citation)}</span>`])}));

  const c1 = card({title: "Property matrix",
    sub: "Green means the measure satisfies the property under randomised numerical probing.",
    right: chip(`seed ${p.seed}`)});
  el.appendChild(c1);
  const pk = p.properties.map(x => x.key);
  c1.body.appendChild(table({raw: true, tall: true, cls: "matrix",
    cols: [{label: "Measure", wrap: true}, {label: "Family"}, ...pk.map(k => ({label: k})),
           {label: "Formula", wrap: true}],
    rows: p.measures.map(m => [
      `<b>${esc(m.label)}</b>`, `<span class="pill">${esc(m.family)}</span>`,
      ...pk.map(k => m[k] ? `<span style="color:var(--good);font-weight:700">✓</span>`
                          : `<span class="muted">·</span>`),
      `<span class="mono" style="font-size:11px">${esc(m.formula)}</span>`])}));
  c1.body.appendChild(h(note(`<b>How to read this.</b> ${esc(p.method)} The derived matrix
    reproduces the published findings: only the odds ratio and Yule's Q/Y are invariant to
    row/column scaling (<span class="mono">O2</span>), and the null-invariant family
    (<span class="mono">O4</span>) is cosine, Jaccard, all-confidence, max-confidence and
    Kulczynski — the measures that do not change when you pad the catalogue with products
    nobody buys together. Support and lift are <em>not</em> null-invariant, which is exactly
    why a lift-only ranking degrades as a catalogue grows.`)));

  const c2 = card({title: "Measure glossary",
    sub: "Every measure computed for every rule, with its formula, range and source."});
  el.appendChild(c2);
  c2.body.appendChild(table({raw: true, tall: true,
    cols: [{label: "Measure"}, {label: "Formula", wrap: true}, {label: "Range", num: true},
           {label: "Family"}, {label: "Source", wrap: true}, {label: "Practitioner note", wrap: true}],
    rows: p.glossary.map(g => [
      `<b>${esc(g.label)}</b>`, `<span class="mono">${esc(g.formula)}</span>`,
      `<span class="mono">${esc(g.range)}</span>`, `<span class="pill">${esc(g.family)}</span>`,
      `<span class="cite">${esc(g.citation)}</span>`, esc(g.note || "")])}));
}

/* ================= deployment ================= */
export async function deployment(el) {
  section(el, "");
  const mc = await A.modelCard();
  el.appendChild(h(`<div class="view-head">
    <p>CRISP-DM Phase 6. A model card (Mitchell et al., FAT* 2019) adapted to an
    unsupervised pattern-mining artefact, plus the provenance needed to reproduce this
    exact rule set.</p></div>`));

  const md = mc.model_details;
  el.appendChild(grid("g2", [
    (() => { const c = card({title: "Model details"});
      c.body.appendChild(kv([
        ["Name", esc(md.name)], ["Version", `<b>${esc(md.version)}</b>`],
        ["Type", esc(md.type)], ["Algorithm", esc(md.algorithm)],
        ["Tuning", esc(md.tuning)],
        ...Object.entries(md.hyperparameters).map(([k, v]) => [k, `<b>${f.sci(v)}</b>`]),
      ])); return c; })(),
    (() => { const c = card({title: "Reproducibility"});
      const pv = md.provenance;
      c.body.appendChild(kv([
        ["Config fingerprint", `<b>${esc(pv.run_fingerprint)}</b>`],
        ["Generated", esc(pv.generated_at_utc)], ["Git SHA", esc(pv.git_sha)],
        ["Python", esc(pv.python)], ["Platform", esc(pv.platform)],
        ["CPU count", esc(pv.cpu_count)],
        ["Data seed", esc(pv.config.data.random_seed)],
        ["Search seed", esc(pv.config.autoresearch.seed)],
      ]));
      c.body.appendChild(h(note(`Every artifact carries this fingerprint — a SHA-256 of the
        full config. Two result sets are comparable only if their fingerprints match.`)));
      return c; })(),
  ]));

  const c1 = card({title: "Intended use"});
  el.appendChild(c1);
  c1.body.appendChild(kv([["Primary", esc(mc.intended_use.primary)],
    ["Users", esc(mc.intended_use.users)]]));
  c1.body.appendChild(h(`<h4 style="font-size:12px;margin:14px 0 6px">Out of scope</h4>
    <ul style="color:var(--text-secondary);padding-left:20px;margin:0">
    ${mc.intended_use.out_of_scope.map(s => `<li>${esc(s)}</li>`).join("")}</ul>`));

  const c2 = card({title: "Performance"});
  el.appendChild(c2);
  c2.body.appendChild(table({raw: true,
    cols: [{label: "Metric"}, {label: "Value", num: true}],
    rows: Object.entries(mc.metrics).filter(([k]) => k !== "objective_terms")
      .map(([k, v]) => [k.replace(/_/g, " "),
        typeof v === "number" ? `<b>${f.sci(v)}</b>` : esc(v)])}));

  el.appendChild(grid("g2", [
    (() => { const c = card({title: "Ethical considerations"});
      c.body.innerHTML = `<ul style="color:var(--text-secondary);padding-left:20px;margin:0">
        ${mc.ethical_considerations.map(s => `<li>${esc(s)}</li>`).join("")}</ul>`; return c; })(),
    (() => { const c = card({title: "Caveats and limitations"});
      c.body.innerHTML = `<ul style="color:var(--text-secondary);padding-left:20px;margin:0">
        ${mc.caveats.map(s => `<li>${esc(s)}</li>`).join("")}</ul>`; return c; })(),
  ]));

  const c3 = card({title: "Known data issues carried into the model",
    sub: "The Phase 2 quality gates that warned, restated here so a consumer of the rules "
       + "sees them without reading the pipeline."});
  el.appendChild(c3);
  c3.body.appendChild(table({raw: true,
    cols: [{label: "Gate"}, {label: "Detail", wrap: true}],
    rows: (mc.data.known_issues || []).map(g => [
      `<span class="mono">${esc(g.gate)}</span>`, esc(g.detail)])}));

  const c4 = card({title: "Service endpoints",
    sub: "The dashboard is a client of the same API a production consumer would use."});
  el.appendChild(c4);
  c4.body.appendChild(table({raw: true,
    cols: [{label: "Method"}, {label: "Path"}, {label: "Returns", wrap: true}],
    rows: [
      ["GET", "<span class='mono'>/api/health</span>", "Liveness plus which artifacts are built"],
      ["GET", "<span class='mono'>/api/rules</span>", "Filtered, sorted, paginated rule set with all 32 measures"],
      ["POST", "<span class='mono'>/api/recommend</span>", "Partial basket → ranked next items with evidence"],
      ["GET", "<span class='mono'>/api/evaluation</span>", "Holdout, significance, sensitivity, benchmark"],
      ["GET", "<span class='mono'>/api/autoresearch</span>", "Full search ledger and objective definition"],
      ["GET", "<span class='mono'>/api/properties</span>", "Derived property matrix and measure glossary"],
      ["GET", "<span class='mono'>/api/model-card</span>", "This page, as JSON"],
    ]}));
}

/* ================= live recommender ================= */
export async function recommender(el) {
  section(el, "");
  const prep = await A.preparation();
  el.appendChild(h(`<div class="view-head">
    <p>The rule set, served. Add items to a basket and the API returns ranked next-item
    suggestions with the evidence attached to each one — the form a recommender service or
    a merchandiser's tool would actually consume.</p></div>`));

  const c = card({title: "Basket",
    sub: "Click products to add them, then see what the rules suggest."});
  el.appendChild(c);
  const chips = h(`<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px"></div>`);
  const picked = new Set();
  const opts = h(`<div style="display:flex;flex-wrap:wrap;gap:6px;max-height:190px;overflow-y:auto"></div>`);
  prep.top_items.forEach(i => {
    const b = h(`<button class="btn">${esc(i.label)}</button>`);
    b.onclick = () => {
      if (picked.has(i.label)) picked.delete(i.label); else picked.add(i.label);
      b.classList.toggle("on", picked.has(i.label));
      render();
    };
    opts.appendChild(b);
  });
  const ctrl = h(`<div class="filters" style="margin:12px 0">
    <div class="field"><label>Only holdout-validated rules</label>
      <div><button class="btn on" id="rc-val">on</button></div></div>
    <div class="field"><label>Min confidence <span class="pill" id="rc-cv">0%</span></label>
      <input id="rc-conf" type="range" min="0" max="0.9" step="0.05" value="0"></div>
  </div>`);
  const out = h(`<div></div>`);
  c.body.appendChild(chips); c.body.appendChild(opts); c.body.appendChild(ctrl); c.body.appendChild(out);

  let validatedOnly = true, minConf = 0;
  ctrl.querySelector("#rc-val").onclick = (e) => {
    validatedOnly = !validatedOnly; e.target.classList.toggle("on", validatedOnly);
    e.target.textContent = validatedOnly ? "on" : "off"; render(); };
  ctrl.querySelector("#rc-conf").oninput = (e) => {
    minConf = +e.target.value; ctrl.querySelector("#rc-cv").textContent = f.pct(minConf, 0); render(); };

  async function render() {
    chips.innerHTML = picked.size
      ? [...picked].map(p => `<span class="chip good">${esc(p)}</span>`).join("")
      : `<span class="muted">Basket is empty — pick a product below.</span>`;
    if (!picked.size) { out.innerHTML = ""; return; }
    out.innerHTML = `<div class="loading">scoring…</div>`;
    try {
      const r = await A.recommend({items: [...picked], top_k: 12,
        validated_only: validatedOnly, min_confidence: minConf});
      if (!r.recommendations.length) {
        out.innerHTML = `<div class="note warn">No rule in the champion set fires on this
          basket. That is a real answer, not a failure: the rule set covers
          ${f.pct((await A.evaluation()).holdout.coverage)} of holdout baskets, and a
          cross-sell surface should stay blank rather than invent a suggestion.</div>`;
        return;
      }
      out.innerHTML = `<h4 style="font-size:12px;margin:6px 0 4px">
        ${r.n_rules_fired} rule${r.n_rules_fired === 1 ? "" : "s"} fired →
        ${r.n_candidates} candidate item${r.n_candidates === 1 ? "" : "s"}</h4>`;
      r.recommendations.forEach((x, i) => out.appendChild(h(`<div class="reco">
        <span class="rank">${i + 1}</span>
        <span class="name">${esc(x.item)}</span>
        <span class="ev">conf <b>${f.pct(x.confidence, 1)}</b> · lift <b>${f.f(x.lift, 2)}</b>
          · holdout conf <b>${x.holdout_confidence != null ? f.pct(x.holdout_confidence, 1) : "–"}</b>
          · p ${f.p(x.p_fisher)}</span>
        ${x.holdout_validated ? `<span class="chip good">validated</span>` : `<span class="chip warn">in-sample only</span>`}
      </div>`)));
      out.appendChild(h(`<div class="note">${esc(r.note)}</div>`));
    } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
  }
  await render();
}
