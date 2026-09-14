/* Dashboard views. Each view is a pure function of API data -> DOM. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const fmt = CH.fmt;

const state = { view: "overview", cache: {}, run: null, study: null, chat: [], busy: false };

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* keep statusText */ }
    throw new Error(msg);
  }
  return r.json();
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const num = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "–");
// counts are integers: 96 should read "96", not "96.00"
const cnt = (v) => (Number.isFinite(v) ? Math.round(v).toLocaleString() : "–");
const pct = (v, d = 1) => (Number.isFinite(v) ? (v * 100).toFixed(d) + "%" : "–");
const dur = (s) => !Number.isFinite(s) ? "–"
  : s < 90 ? `${s.toFixed(0)}s` : s < 5400 ? `${(s / 60).toFixed(1)} min` : `${(s / 3600).toFixed(1)} h`;
const when = (ts) => (ts ? new Date(ts * 1000).toLocaleString(undefined,
  { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "–");

const stat = (label, value, sub, unit) => `
  <div class="stat"><div class="stat-label">${esc(label)}</div>
  <div class="stat-value">${value}${unit ? `<small>${esc(unit)}</small>` : ""}</div>
  ${sub ? `<div class="stat-sub">${sub}</div>` : ""}</div>`;

const card = (title, note, body, cls = "") => `
  <div class="card ${cls}"><h3>${esc(title)}</h3>
  ${note ? `<p class="note">${note}</p>` : ""}${body}</div>`;

const chartBox = (id, h = 210) => `<div id="${id}" style="min-height:${h}px"></div>`;

function table(cols, rows) {
  return `<div class="tbl-wrap"><table><thead><tr>${cols.map((c) =>
    `<th class="${c.num ? "num" : ""}">${esc(c.label)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${cols.map((c) =>
      `<td class="${c.num ? "num" : ""} ${c.cls ? c.cls(r) : ""}">${c.render ? c.render(r) : esc(r[c.key])}</td>`
    ).join("")}</tr>`).join("")}</tbody></table></div>`;
}

const statusPill = (s) => {
  const m = { finished: "pill-good", running: "pill-info", failed: "pill-bad",
              diverged: "pill-bad", cancelled: "pill-warn" };
  return `<span class="pill ${m[s] || "pill-muted"}">${esc(s)}</span>`;
};

const VIEWS = {
  overview: { title: "Overview", sub: "Project status across the full CRISP-DM cycle." },
  data: { title: "Data understanding & preparation", sub: "CRISP-DM phases 2–3. Corpus profile, tokenizer, deduplication and the train/validation leakage audit." },
  training: { title: "Modeling", sub: "CRISP-DM phase 4. Optimization curves, throughput, and hardware utilization per run." },
  evaluation: { title: "Evaluation", sub: "CRISP-DM phase 5. Held-out quality, calibration, context use, and decoding behaviour." },
  autoresearch: { title: "AutoResearch", sub: "Automated hill climbing over published techniques — each trial reproduces one paper's claim at our scale." },
  papers: { title: "Paper registry", sub: "Every citation backing an architectural or optimization choice. Metadata verified against the arXiv API." },
  model: { title: "Model card", sub: "CRISP-DM phase 6. Architecture, parameter accounting, and deployment characteristics." },
  chat: { title: "Chat", sub: "The deployed model, served by the same inference path used for offline evaluation." },
};

/* ------------------------------------------------------------------ router */
async function render(view) {
  state.view = view;
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $("#view-title").textContent = VIEWS[view].title;
  $("#view-sub").textContent = VIEWS[view].sub;
  const c = $("#content");
  c.innerHTML = `<div class="empty">loading…</div>`;
  CH.clear();
  try {
    await ({ overview: vOverview, data: vData, training: vTraining, evaluation: vEval,
             autoresearch: vAuto, papers: vPapers, model: vModel, chat: vChat }[view])(c);
  } catch (e) {
    c.innerHTML = `<div class="empty">${esc(e.message)}<br><span class="dim" style="font-size:12px">
      Some views need a completed pipeline stage. See the README quickstart.</span></div>`;
  }
}

/* ---------------------------------------------------------------- overview */
async function vOverview(c) {
  const o = await api("/api/overview");
  const bp = o.best_pretrain?.summary || {};
  const bs = o.best_sft?.summary || {};
  const study = (o.studies || [])[0]?.summary || {};
  const corpus = o.corpus || {};

  const phases = [
    ["1 · Business understanding", "docs/1_business_understanding.md", true],
    ["2 · Data understanding", "profile", !!corpus.train_tokens],
    ["3 · Data preparation", "tokens", !!corpus.train_tokens],
    ["4 · Modeling", "runs", (o.n_runs || 0) > 0],
    ["5 · Evaluation", "eval", !!bp.val_ppl || !!bp.best_val_ppl],
    ["6 · Deployment", "chat", !!o.best_sft || !!o.best_pretrain],
  ];

  c.innerHTML = `
    <div class="grid cols-4">
      ${stat("Best pretrain perplexity", num(bp.best_val_ppl, 2),
        bp.best_val_loss ? `val loss ${num(bp.best_val_loss, 4)} · ${esc(o.best_pretrain?.name || "")}` : "no pretrain run yet")}
      ${stat("Chat model perplexity", num(bs.best_val_ppl, 2),
        bs.best_val_loss ? `SFT val loss ${num(bs.best_val_loss, 4)}` : "no SFT run yet")}
      ${stat("Parameters", fmt(bp.n_params_total || 0),
        `${fmt(bp.n_params_nonemb || 0)} non-embedding`)}
      ${stat("Tokens trained", fmt(o.total_tokens_trained || 0),
        `${o.n_runs || 0} runs · ${dur(o.total_gpu_seconds)} GPU time`)}
    </div>

    <div class="section-title">Pipeline</div>
    <div class="grid cols-3">
      ${card("CRISP-DM phase status", "Each phase produces a durable artifact; the dashboard reads those artifacts rather than recomputing them.",
        `<div style="display:flex;flex-direction:column;gap:6px">${phases.map(([n, , ok]) =>
          `<div style="display:flex;justify-content:space-between;align-items:center;font-size:12.5px">
            <span>${esc(n)}</span>${ok ? '<span class="pill pill-good">complete</span>'
                                        : '<span class="pill pill-muted">pending</span>'}</div>`).join("")}</div>`)}
      ${card("Corpus", "Prepared token stream backing every run.",
        `<dl class="kv">
          <dt>train tokens</dt><dd>${fmt(corpus.train_tokens || 0)}</dd>
          <dt>val tokens</dt><dd>${fmt(corpus.val_tokens || 0)}</dd>
          <dt>vocabulary</dt><dd>${corpus.vocab_size || "–"}</dd>
          <dt>bytes / token</dt><dd>${num(corpus.bytes_per_token_train, 2)}</dd>
        </dl>`)}
      ${card("AutoResearch", "Greedy hill climb over the paper registry.",
        study.champion_mean ? `<dl class="kv">
          <dt>baseline</dt><dd>${num(study.baseline_mean, 4)}</dd>
          <dt>champion</dt><dd>${num(study.champion_mean, 4)}</dd>
          <dt>improvement</dt><dd class="neg">${num(study.improvement, 4)} (${num(study.improvement_pct, 2)}%)</dd>
          <dt>effect size</dt><dd>d = ${num(study.effect_size_cohens_d, 2)}</dd>
          <dt>accepted</dt><dd>${(study.accepted || []).length} of ${study.n_tested || 0} tested</dd>
        </dl>` : `<p class="claim">${(o.studies || []).length ? "study in progress — check the AutoResearch tab." : "no study run yet."}</p>`)}
    </div>

    <div class="section-title">Literature coverage</div>
    <div class="grid cols-4">
      ${stat("Papers in registry", o.n_papers, "arXiv-verified metadata")}
      ${stat("Testable interventions", o.n_interventions, "each mapped to its source paper")}
      ${stat("Runs recorded", o.n_runs || 0,
        Object.entries(o.runs_by_phase || {}).map(([k, v]) => `${v} ${k}`).join(" · "))}
      ${stat("Peak MFU", pct(bp.mfu_peak ?? null), "model FLOPs utilization")}
    </div>

    <div class="section-title">Recent runs</div>
    <div class="card">${table([
      { label: "Run", key: "run_id", render: (r) => `<a href="#" data-run="${esc(r.run_id)}" class="mono">${esc(r.run_id)}</a>` },
      { label: "Name", key: "name" },
      { label: "Phase", key: "phase", render: (r) => `<span class="pill pill-muted">${esc(r.phase)}</span>` },
      { label: "Status", key: "status", render: (r) => statusPill(r.status) },
      { label: "Val loss", num: true, render: (r) => num(r.summary?.best_val_loss, 4) },
      { label: "Perplexity", num: true, render: (r) => num(r.summary?.best_val_ppl, 2) },
      { label: "Params", num: true, render: (r) => fmt(r.n_params || 0) },
      { label: "Duration", num: true, render: (r) => dur(r.summary?.elapsed_s) },
      { label: "Started", render: (r) => when(r.created_at) },
    ], o.recent_runs || [])}</div>`;

  $$("[data-run]").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault(); state.run = a.dataset.run; render("training");
  }));
}

/* -------------------------------------------------------------------- data */
async function vData(c) {
  const d = await api("/api/data-profile");
  let sft = null;
  try { sft = await api("/api/sft-profile"); } catch { /* SFT stage optional */ }
  const tr = d.profile_train || {}, va = d.profile_val || {};
  const tk = d.tokenizer || {}, meta = d.meta || {};
  const ctl = d.dedup_control || {}, lk = d.leakage_audit || {}, nd = d.near_dedup_train || {},
        ex = d.exact_dedup_train || {};

  const cleanFinding = (ex.n_exact_dupes === 0 && nd.n_removed === 0 && lk.n_leaked === 0);

  c.innerHTML = `
    <div class="grid cols-4">
      ${stat("Documents", fmt(tr.n_docs || 0), `${fmt(va.n_docs || 0)} held out`)}
      ${stat("Train tokens", fmt(meta.train_tokens || 0), `${fmt(meta.val_tokens || 0)} validation`)}
      ${stat("Compression", num(tk.bytes_per_token, 2), "bytes per token", "B/tok")}
      ${stat("Vocabulary", meta.vocab_size || "–", `${pct(tr.vocab_coverage)} observed in sample`)}
    </div>

    <div class="section-title">Document & token distributions</div>
    <div class="grid cols-2">
      ${card("Document length (characters)",
        `Median ${fmt(tr.char_len?.p50)} chars, p95 ${fmt(tr.char_len?.p95)}. Clipped at p99.5.`,
        chartBox("h-char", 190))}
      ${card("Document length (tokens)",
        `Median ${fmt(tr.token_len?.p50)} tokens. The training context is ${meta.vocab_size ? "512" : "–"} tokens, so most documents fit whole.`,
        chartBox("h-tok", 190))}
    </div>

    <div class="grid cols-2">
      ${card("Token frequency (Zipf)",
        "Rank vs. frequency on log–log axes. A near-straight line is the expected Zipfian signature of natural text; deviation at the tail shows where the vocabulary over-fits this corpus.",
        chartBox("zipf", 220))}
      ${card("Tokenizer merge saturation",
        "Occurrence count of each BPE merge as it was learned. When this floor approaches zero the vocabulary is larger than the corpus can support — visible evidence for choosing a vocabulary size.",
        chartBox("merges", 220))}
    </div>

    <div class="section-title">Data quality</div>
    <div class="grid cols-2">
      ${card("Deduplication & leakage audit",
        "Exact hashing, then MinHash/LSH near-duplicate clustering, then a train↔validation leakage check.",
        `<dl class="kv">
          <dt>documents in</dt><dd>${cnt(ex.n_in || 0)}</dd>
          <dt>exact duplicates</dt><dd>${cnt(ex.n_exact_dupes || 0)}</dd>
          <dt>near duplicates</dt><dd>${cnt(nd.n_removed || 0)} in ${cnt(nd.n_clusters || 0)} clusters</dd>
          <dt>LSH candidate pairs</dt><dd>${cnt(nd.n_lsh_candidates || 0)}</dd>
          <dt>MinHash config</dt><dd>${nd.bands || "–"} bands × ${nd.rows_per_band || "–"} rows, J ≥ ${nd.threshold ?? "–"}</dd>
          <dt>val docs leaked</dt><dd>${cnt(lk.n_leaked || 0)} of ${cnt(lk.n_val_in || 0)} (${pct(lk.leak_rate, 3)})</dd>
        </dl>
        ${cleanFinding ? `<p class="claim" style="margin-top:10px">
          <span class="pill pill-good">clean</span> No duplicates or leakage found. This corpus is
          model-generated and was already deduplicated upstream — the finding is credible only because
          the detectors were validated on planted duplicates first (right).</p>` : ""}`)}
      ${card("Detector positive control",
        "A zero from a detector is indistinguishable from a broken detector. Before auditing the real corpus we plant known duplicates in a clean sample and measure recall.",
        ctl.ran ? `${table([
          { label: "Detector", key: "k" },
          { label: "Planted", key: "p", num: true },
          { label: "Detected", key: "d", num: true },
          { label: "Recall", key: "r", num: true },
          { label: "False pos.", key: "f", num: true },
        ], [
          { k: "Exact (SHA-1)", p: ctl.exact_planted, d: ctl.exact_detected, r: pct(ctl.exact_recall, 0), f: "0" },
          { k: "Near (MinHash/LSH)", p: ctl.near_planted, d: ctl.near_detected, r: pct(ctl.near_recall, 0), f: ctl.near_false_positives },
          { k: "Train↔val leakage", p: ctl.leak_planted, d: ctl.leak_detected, r: pct(ctl.leak_recall, 0), f: ctl.leak_false_positives },
        ])}
        <p class="claim" style="margin-top:9px">Control run on ${cnt(ctl.n_control_docs)} held-back documents.</p>`
        : `<p class="claim">control not run</p>`)}
    </div>

    <div class="grid cols-2">
      ${card("Character composition", "Share of characters by class in the training sample.",
        chartBox("charcls", 150))}
      ${card("Most frequent tokens", "Top of the unigram distribution, with the decoded surface form.",
        `<div class="tbl-wrap" style="max-height:260px;overflow-y:auto">${table([
          { label: "#", key: "rank", num: true },
          { label: "Token", key: "piece", render: (r) => `<code>${esc(r.piece.replace(/ /g, "␣").replace(/\n/g, "\\n"))}</code>` },
          { label: "Count", key: "count", num: true, render: (r) => r.count.toLocaleString() },
          { label: "Share", key: "freq", num: true, render: (r) => pct(r.freq, 2) },
        ], (tr.top_tokens || []).slice(0, 25))}</div>`)}
    </div>

    ${sft ? `<div class="section-title">Instruction-tuning data</div>
    <div class="grid cols-4">
      ${stat("Chat examples", fmt(sft.train?.n || 0), `${fmt(sft.val?.n || 0)} validation`)}
      ${stat("Supervised tokens", pct(sft.train?.supervised_fraction), "assistant turns only")}
      ${stat("Padding", pct(sft.train?.padding_fraction), `packed to ${sft.train?.max_len} tokens`)}
      ${stat("Mean length", fmt(sft.train?.mean_len || 0), `p95 ${fmt(sft.train?.p95_len || 0)} tokens`)}
    </div>
    <div class="card" style="margin-top:14px"><h3>Rendered training example</h3>
      <p class="note">ChatML. Gradients flow only through assistant content and its closing <code>&lt;|im_end|&gt;</code>.</p>
      <pre>${esc((sft.examples || [""])[0])}</pre></div>` : ""}`;

  CH.hist($("#h-char"), { edges: tr.char_len_hist?.edges, counts: tr.char_len_hist?.counts, xLabel: "characters" });
  CH.hist($("#h-tok"), { edges: tr.token_len_hist?.edges, counts: tr.token_len_hist?.counts, xLabel: "tokens" });
  CH.line($("#zipf"), {
    series: [{ name: "token frequency", points: (tr.zipf || []).map((z) => [z.rank, z.count]) }],
    xScale: "log", yScale: "log", xLabel: "rank", yLabel: "count", height: 220,
  });
  const ml = tk.merge_log_head || [];
  CH.line($("#merges"), {
    series: [{ name: "merge count", points: ml.map((m) => [m.rank + 1, m.count]) }],
    yScale: "log", xLabel: "merge rank", yLabel: "pair count", height: 220,
  });
  const cc = tr.char_classes || {};
  CH.bars($("#charcls"), {
    horizontal: true, rowH: 24, labelW: 74, height: 150,
    items: Object.entries(cc).map(([k, v]) => ({ label: k, value: v })),
    valueFmt: (v) => pct(v, 1),
  });
}

/* ---------------------------------------------------------------- training */
async function vTraining(c) {
  const runs = await api("/api/runs");
  if (!runs.length) { c.innerHTML = `<div class="empty">no runs yet — train one first</div>`; return; }
  const trainable = runs.filter((r) => r.phase !== "autoresearch");
  const list = trainable.length ? trainable : runs;
  if (!state.run || !list.find((r) => r.run_id === state.run)) state.run = list[0].run_id;
  const d = await api(`/api/runs/${state.run}`);
  const m = d.metrics || {}, s = d.summary || {}, cfg = d.config || {}, env = d.env || {};
  const budget = (d.events || []).find((e) => e.message === "budget")?.data || {};

  const S = (k) => m[k] || [];
  const lastOf = (k) => (S(k).length ? S(k).at(-1)[1] : null);
  const minOf = (k) => (S(k).length ? S(k).reduce((a, p) => (p[1] < a[1] ? p : a)) : null);
  // A running run has no summary yet; fall back to the live metric stream.
  const live = d.status === "running" || !s.best_val_loss;
  const bestPt = minOf("val_loss");
  const kBest = live && bestPt ? bestPt[1] : s.best_val_loss;
  const kBestPpl = live && bestPt ? Math.exp(Math.min(bestPt[1], 20)) : s.best_val_ppl;
  const kBestStep = live && bestPt ? bestPt[0] : s.best_step;
  const kTps = live ? lastOf("tokens_per_s") : s.tokens_per_s;
  const kTokens = live ? lastOf("tokens_seen") : s.tokens_seen;
  const kElapsed = live ? lastOf("elapsed_s") : s.elapsed_s;

  c.innerHTML = `
    <div class="controls">
      <label>Run
        <select id="run-sel">${list.map((r) => `<option value="${esc(r.run_id)}" ${r.run_id === state.run ? "selected" : ""}>
          ${esc(r.name)} · ${esc(r.phase)} · ${esc(r.run_id)}</option>`).join("")}</select>
      </label>
      ${statusPill(d.status)}
      <span class="pill pill-muted">${esc(env.device || "?")}</span>
      <span class="pill pill-muted">${esc(env.chip || "")}</span>
    </div>

    <div class="grid cols-4">
      ${stat("Best val loss", num(kBest, 4), `perplexity ${num(kBestPpl, 2)} @ step ${kBestStep ?? "–"}${live ? " · live" : ""}`)}
      ${stat("Throughput", fmt(kTps || 0), live ? "tokens / second (latest)" : "tokens / second (run mean)", "tok/s")}
      ${stat("Peak MFU", pct(Math.max(0, ...S("mfu").map((p) => p[1])) || null),
        `vs ${num((env.peak_flops_measured || 0) / 1e12, 2)} TFLOP/s measured roofline`)}
      ${stat("Wall time", dur(kElapsed), `${fmt(kTokens || 0)} tokens seen · step ${lastOf("tokens_seen") !== null ? S("tokens_seen").at(-1)[0] : "–"} of ${cfg.train?.steps ?? "–"}`)}
    </div>

    <div class="section-title">Optimization</div>
    <div class="grid cols-2">
      ${card("Loss", "Training loss is a running minibatch estimate; validation and train-holdout are measured on fixed batches, so the gap between them is a real generalization gap.",
        chartBox("c-loss", 240))}
      ${card("Generalization gap", "Validation minus train-holdout, both on identical fixed batches.",
        chartBox("c-gap", 240))}
    </div>
    <div class="grid cols-3">
      ${card("Learning rate", `${esc(cfg.train?.schedule || "")} schedule, ${cfg.train?.warmup_steps ?? "–"} warmup steps.`, chartBox("c-lr", 180))}
      ${card("Gradient norm", `Clipped at ${cfg.train?.grad_clip ?? "–"}. Spikes here precede loss divergence.`, chartBox("c-gn", 180))}
      ${card("Throughput & utilization", "Tokens/s per logged step.", chartBox("c-tps", 180))}
    </div>

    <div class="section-title">Compute budget</div>
    <div class="grid cols-2">
      ${card("Token budget accounting",
        "Stated before training, not discovered after. Chinchilla-optimal is ≈20 tokens per non-embedding parameter.",
        `<dl class="kv">
          <dt>non-embedding params</dt><dd>${fmt(budget.n_params_nonemb || 0)}</dd>
          <dt>tokens this run</dt><dd>${fmt(budget.total_train_tokens || 0)}</dd>
          <dt>epochs over corpus</dt><dd>${num(budget.epochs_over_corpus, 2)}×</dd>
          <dt>Chinchilla-optimal</dt><dd>${fmt(budget.chinchilla_optimal_tokens || 0)} tokens</dd>
          <dt>ratio to optimal</dt><dd>${num(budget.chinchilla_ratio, 2)}×</dd>
          <dt>FLOPs / token</dt><dd>${fmt(budget.flops_per_token || 0)}</dd>
          <dt>total FLOPs</dt><dd>${fmt(budget.total_flops || 0)}</dd>
        </dl>
        ${budget.chinchilla_ratio < 1 ? `<p class="claim" style="margin-top:9px">
          <span class="pill pill-warn">under-trained</span> At ${num(budget.chinchilla_ratio, 2)}× the
          compute-optimal token count this model is data-limited, not capacity-limited: loss would keep
          falling with more tokens at fixed size.</p>` : ""}`)}
      ${card("Memory", "Allocated and driver-reserved device memory over the run.", chartBox("c-mem", 200))}
    </div>

    ${d.samples?.length ? `<div class="section-title">Generation samples during training</div>
      <div class="card">${table([
        { label: "Step", key: "step", num: true },
        { label: "Prompt", key: "prompt", render: (r) => `<span class="mono">${esc(r.prompt)}</span>` },
        { label: "Completion", key: "completion", render: (r) => esc(r.completion).slice(0, 400) },
      ], d.samples.slice(-8).reverse())}</div>` : ""}

    <div class="section-title">Configuration</div>
    <div class="grid cols-2">
      ${card("Model", "", `<pre>${esc(JSON.stringify(cfg.model || {}, null, 1))}</pre>`)}
      ${card("Training & environment", "", `<pre>${esc(JSON.stringify({ ...(cfg.train || {}), _env: env }, null, 1))}</pre>`)}
    </div>`;

  $("#run-sel").addEventListener("change", (e) => { state.run = e.target.value; render("training"); });

  CH.line($("#c-loss"), {
    series: [
      { name: "train (minibatch)", points: S("train_loss"), faint: true },
      { name: "validation", points: S("val_loss") },
      { name: "train holdout", points: S("train_holdout_loss") },
    ], xLabel: "step", yLabel: "cross-entropy (nats)", height: 240,
  });
  CH.line($("#c-gap"), {
    series: [{ name: "val − train holdout", points: S("generalization_gap"), color: CH.palette().series[1] }],
    xLabel: "step", yLabel: "nats", height: 240, yZero: true,
  });
  CH.line($("#c-lr"), { series: [{ name: "learning rate", points: S("lr") }], xLabel: "step", height: 180 });
  CH.line($("#c-gn"), { series: [{ name: "grad norm", points: S("grad_norm"), color: CH.palette().series[1] }], xLabel: "step", height: 180 });
  CH.line($("#c-tps"), { series: [{ name: "tokens/s", points: S("tokens_per_s"), color: CH.palette().series[2] }], xLabel: "step", height: 180 });
  CH.line($("#c-mem"), {
    series: [{ name: "allocated (MB)", points: S("allocated_mb") },
             { name: "driver peak (MB)", points: S("peak_mb") }],
    xLabel: "step", height: 200,
  });
}

/* -------------------------------------------------------------- evaluation */
async function vEval(c) {
  const runs = await api("/api/runs");
  const withEval = [];
  for (const r of runs.slice(0, 25)) {
    const d = await api(`/api/runs/${r.run_id}`).catch(() => null);
    if (d?.eval) withEval.push(d);
  }
  if (!withEval.length) {
    c.innerHTML = `<div class="empty">No evaluation report yet.<br>
      <span class="dim" style="font-size:12px">Run <code>python -m slm.evaluate</code> to produce one.</span></div>`;
    return;
  }
  const d = withEval.find((x) => x.run_id === state.run) || withEval[0];
  const ev = d.eval, ho = ev.heldout || {}, lat = ev.latency || {}, inst = ev.instruction;

  c.innerHTML = `
    <div class="controls">
      <label>Run <select id="ev-sel">${withEval.map((r) =>
        `<option value="${esc(r.run_id)}" ${r.run_id === d.run_id ? "selected" : ""}>
          ${esc(r.name)} · ${esc(r.phase)}</option>`).join("")}</select></label>
    </div>
    <div class="grid cols-4">
      ${stat("Perplexity", num(ho.ppl, 2), `cross-entropy ${num(ho.loss, 4)} nats`)}
      ${stat("Bits per byte", num(ho.bits_per_byte, 3), "tokenizer-independent unit")}
      ${stat("Top-1 accuracy", pct(ho.acc_top1), `top-5 ${pct(ho.acc_top5)}`)}
      ${stat("Calibration error", num(ho.ece, 4), "expected calibration error (ECE)")}
    </div>

    <div class="section-title">Where the loss comes from</div>
    <div class="grid cols-2">
      ${card("Reliability diagram",
        `Bars are observed accuracy per confidence bin; the line is perfect calibration. Bars below the line mean the model is overconfident. ECE = ${num(ho.ece, 4)}.`,
        chartBox("c-rel", 210))}
      ${card("Loss by token-frequency decile",
        "Targets bucketed by corpus frequency mass (decile 1 = most common tokens). The aggregate perplexity hides how much worse rare tokens are.",
        chartBox("c-freq", 210))}
    </div>
    <div class="grid cols-2">
      ${card("Loss by position in context",
        "Average loss at each position. A falling curve means the model genuinely uses earlier context; a flat one means it is effectively ignoring it.",
        chartBox("c-pos", 210))}
      ${card("Inference performance",
        `Measured on ${esc(String(lat.device || d.env?.device || ""))} with a KV cache.`,
        `<dl class="kv">
          <dt>decode speed</dt><dd>${num(lat.decode_tok_per_s, 1)} tok/s</dd>
          <dt>time to first token</dt><dd>${num(lat.ttft_ms, 1)} ms</dd>
          <dt>KV cache (full context)</dt><dd>${num(lat.kv_cache_mb_full_ctx, 2)} MB</dd>
          <dt>parameters</dt><dd>${fmt(lat.params_total || 0)}</dd>
        </dl>`)}
    </div>

    <div class="section-title">Decoding study</div>
    <div class="card">
      <p class="note">A direct test of the degeneration result (Holtzman et al. 2019): maximization
      decoding collapses into repetition, while nucleus sampling does not. Same prompts, same seed,
      only the decoding rule changes.</p>
      ${chartBox("c-dec", 200)}
      <hr class="hairline">
      ${table([
        { label: "Strategy", key: "name" },
        { label: "Repeated 4-grams", key: "repeated_4gram_rate", num: true,
          render: (r) => pct(r.repeated_4gram_rate, 1),
          cls: (r) => (r.repeated_4gram_rate > 0.2 ? "pos" : "") },
        { label: "distinct-1", key: "distinct_1", num: true, render: (r) => num(r.distinct_1, 3) },
        { label: "distinct-2", key: "distinct_2", num: true, render: (r) => num(r.distinct_2, 3) },
        { label: "Stop rate", key: "stop_rate", num: true, render: (r) => pct(r.stop_rate, 0) },
        { label: "Mean words", key: "mean_words", num: true },
        { label: "tok/s", key: "tok_per_s", num: true },
      ], ev.decoding || [])}
      <hr class="hairline">
      <h3 style="font-size:12.5px">Sample completions</h3>
      ${(ev.decoding || []).slice(0, 3).map((x) => `
        <p class="claim" style="margin:8px 0 2px"><span class="pill pill-muted">${esc(x.name)}</span></p>
        <pre>${esc(x.sample)}</pre>`).join("")}
    </div>

    ${inst ? `<div class="section-title">Instruction compliance</div>
    <div class="grid cols-4">
      ${stat("Word inclusion", pct(inst.word_inclusion_rate), `across ${inst.n_examples} requests`)}
      ${stat("Fully compliant", pct(inst.fully_compliant_rate), "all requested words used")}
      ${stat("distinct-2", num(inst.distinct_2, 3), "lexical diversity of replies")}
      ${stat("Repeated 4-grams", pct(inst.repeated_4gram_rate, 1), "lower is better")}
    </div>
    <div class="card" style="margin-top:14px">
      <h3>Sampled responses</h3>
      <p class="note">The user asks for a story containing specific words; compliance is checked automatically.</p>
      ${table([
        { label: "Requested words", key: "words", render: (r) => `<code>${esc((r.words || []).join(", "))}</code>` },
        { label: "Used", key: "found", num: true, render: (r) => `${r.found}/${r.n}` },
        { label: "Reply", key: "reply", render: (r) => esc(r.reply) },
      ], inst.examples || [])}
    </div>` : ""}`;

  $("#ev-sel").addEventListener("change", (e) => { state.run = e.target.value; render("evaluation"); });

  CH.reliability($("#c-rel"), ho.reliability || []);
  CH.bars($("#c-freq"), {
    horizontal: false, height: 210, xLabel: "frequency decile (1 = most common)",
    items: (ho.loss_by_freq_decile || []).map((r) => ({
      label: String(r.decile), value: r.loss,
      tip: `<div class="tt-title">decile ${r.decile}</div>
            <div class="tt-row"><span>loss</span><b>${num(r.loss, 3)}</b></div>
            <div class="tt-row"><span>tokens</span><b>${r.n.toLocaleString()}</b></div>` })),
    valueFmt: (v) => num(v, 2),
  });
  CH.line($("#c-pos"), {
    series: [{ name: "loss at position", points: (ho.loss_by_position || []).map((v, i) => [i, v]) }],
    xLabel: "position in context", yLabel: "nats", height: 210,
  });
  const dec = ev.decoding || [];
  CH.bars($("#c-dec"), {
    horizontal: true, rowH: 24, labelW: 150, height: Math.max(120, dec.length * 24 + 48),
    items: dec.map((x) => ({
      label: x.name, value: x.repeated_4gram_rate,
      color: x.repeated_4gram_rate > 0.2 ? CH.palette().critical : CH.palette().series[0],
      tip: `<div class="tt-title">${esc(x.name)}</div>
            <div class="tt-row"><span>repeated 4-grams</span><b>${pct(x.repeated_4gram_rate, 1)}</b></div>
            <div class="tt-row"><span>distinct-2</span><b>${num(x.distinct_2, 3)}</b></div>` })),
    valueFmt: (v) => pct(v, 0), xLabel: "repeated 4-gram rate",
  });
}

/* ------------------------------------------------------------ autoresearch */
async function vAuto(c) {
  const studies = await api("/api/studies");
  if (!studies.length) {
    c.innerHTML = `<div class="empty">No study yet.<br>
      <span class="dim" style="font-size:12px">Run <code>python -m slm.autoresearch</code>.</span></div>`;
    return;
  }
  if (!state.study || !studies.find((s) => s.study === state.study)) state.study = studies[0].study;
  const d = await api(`/api/studies/${encodeURIComponent(state.study)}`);
  const sum = d.summary || {}, cfg = d.config || {}, trials = d.trials || [];
  const running = d.status !== "finished";

  const noise = trials.filter((t) => t.intervention === "__baseline__");
  const sigma = sum.noise_sigma ?? (noise.length > 1 ? stdev(noise.map((t) => t.val_loss)) : 0);
  const thresh = sum.accept_threshold ?? Math.max(cfg.min_delta || 0.005, (cfg.z_accept || 1) * sigma);
  const cand = trials.filter((t) => !t.intervention.startsWith("__"));
  const rounds = [...new Set(cand.map((t) => t.round))].sort((a, b) => a - b);

  // hill-climb trajectory: incumbent loss after each accepted move
  const incs = trials.filter((t) => t.intervention === "__incumbent__")
    .sort((a, b) => a.round - b.round);
  const traj = [[0, sum.baseline_mean ?? (noise.length ? mean(noise.map((t) => t.val_loss)) : null)]];
  const accepted = cand.filter((t) => t.accepted).sort((a, b) => a.round - b.round);
  accepted.forEach((t, i) => traj.push([i + 1, t.val_loss]));

  c.innerHTML = `
    <div class="controls">
      <label>Study <select id="st-sel">${studies.map((s) =>
        `<option ${s.study === state.study ? "selected" : ""}>${esc(s.study)}</option>`).join("")}</select></label>
      ${statusPill(d.status)}
      ${running ? `<span class="pill pill-info">${cand.length} trials so far</span>` : ""}
    </div>

    <div class="grid cols-4">
      ${stat("Baseline → champion", `${num(sum.baseline_mean, 3)} → ${num(sum.champion_mean, 3)}`,
        sum.improvement ? `<span class="neg">${num(sum.improvement, 4)} nats (${num(sum.improvement_pct, 2)}%)</span>` : "in progress")}
      ${stat("Effect size", sum.effect_size_cohens_d != null ? `d = ${num(sum.effect_size_cohens_d, 2)}` : "–",
        "champion vs baseline across seeds")}
      ${stat("Noise floor", `σ = ${num(sigma, 4)}`, `accept threshold ${num(thresh, 4)} nats`)}
      ${stat("Trials run", cand.length, `${accepted.length} accepted · ${rounds.length} rounds · ${dur((sum.wall_minutes || 0) * 60)}`)}
    </div>

    <div class="section-title">Method</div>
    <div class="card">
      <p class="claim">The search starts from a deliberately dated baseline — LayerNorm, learned absolute
      positions, a GELU feed-forward, full multi-head attention, untied embeddings, dropout 0.1, AdamW with
      a cosine schedule — and greedily adds one published technique per round. Every candidate is trained
      as a paired trial: same seed, same fixed evaluation batches, same proxy scale
      (${cfg.proxy?.model?.n_layer ?? "–"} layers × d${cfg.proxy?.model?.d_model ?? "–"},
      ${cfg.proxy?.train?.steps ?? "–"} steps). A move is accepted only if it beats the incumbent by more
      than <b>${num(thresh, 4)}</b> nats, which is ${cfg.z_accept ?? 1}σ of the baseline's own seed-to-seed
      variance measured over ${cfg.n_seeds ?? "–"} runs. Anything smaller is indistinguishable from luck.</p>
      <p class="claim" style="margin-top:8px"><b>Known limitation:</b> greedy hill climbing cannot discover
      techniques that only pay off in combination, and the accepted order is itself an artifact of the
      search path. Rejections below mean “did not clear the bar at this scale, in this order” — not
      “the paper is wrong.”</p>
      ${sum.converged === false && (sum.still_significant || []).length ? `
      <p class="claim" style="margin-top:8px"><span class="pill pill-warn">budget-limited</span>
      The search stopped at its round limit, not because candidates ran out:
      ${sum.still_significant.map((k) => `<code>${esc(k)}</code>`).join(", ")} still cleared the
      significance bar in the final round and would most likely have been accepted with more rounds.
      The champion below is therefore a lower bound on what the search space contains.</p>` : ""}
    </div>

    <div class="section-title">Hill-climb trajectory</div>
    <div class="grid cols-2">
      ${card("Validation loss after each accepted move",
        "Step 0 is the baseline mean across seeds. The shaded band is ±1σ of baseline seed noise.",
        chartBox("c-traj", 230))}
      ${card("Accepted moves, in order",
        "Each row is a technique that cleared the significance bar, with the paper that proposed it.",
        accepted.length ? `<div style="display:flex;flex-direction:column;gap:9px">${accepted.map((t, i) => `
          <div style="display:flex;gap:9px;align-items:flex-start">
            <span class="pill pill-good">${i + 1}</span>
            <div style="min-width:0">
              <div style="font-size:12.8px;font-weight:560">${esc(t.meta?.title || t.intervention)}</div>
              <div class="claim">${esc(t.meta?.claim || "")}</div>
              <div class="dim mono" style="font-size:11px">Δ ${num(t.delta, 4)} nats · z = ${num(t.z_score, 2)} ·
                ${(t.meta?.paper_records || []).map((p) => esc(p.authors + " " + p.year)).join("; ")}</div>
            </div></div>`).join("")}</div>`
          : `<p class="claim">no move has cleared the threshold yet</p>`)}
    </div>

    ${rounds.map((rn) => {
      const rt = cand.filter((t) => t.round === rn).sort((a, b) => (a.delta ?? 9) - (b.delta ?? 9));
      const done = rt.filter((t) => Number.isFinite(t.delta));
      const inc = incs.find((t) => t.round === rn);
      return `<div class="section-title">Round ${rn} — incumbent ${num(inc?.val_loss, 4)}</div>
        <div class="card">
          <p class="note">Change in validation loss versus the incumbent. Negative is better. Bars inside
          the grey band are within seed noise and are <b>not</b> treated as evidence.</p>
          ${chartBox(`c-r${rn}`, Math.max(120, done.length * 24 + 52))}
          <hr class="hairline">
          ${table([
            { label: "Intervention", key: "intervention",
              render: (t) => `<b>${esc(t.meta?.title || t.intervention)}</b>` },
            { label: "Δ loss", key: "delta", num: true, render: (t) => num(t.delta, 4),
              cls: (t) => (t.delta < 0 ? "neg" : "pos") },
            { label: "z", key: "z_score", num: true, render: (t) => num(t.z_score, 2) },
            { label: "Verdict", key: "accepted", render: (t) =>
              !Number.isFinite(t.delta) ? '<span class="pill pill-info">running</span>'
              : t.accepted ? '<span class="pill pill-good">accepted</span>'
                : (t.status === "diverged" || t.val_loss > 1e9 ? '<span class="pill pill-bad">diverged</span>'
                  : (t.delta < -thresh ? '<span class="pill pill-warn">significant, not best</span>'
                    : (t.delta > thresh ? '<span class="pill pill-bad">worse</span>'
                      : '<span class="pill pill-muted">within noise</span>'))) },
            { label: "tok/s", key: "tokens_per_s", num: true, render: (t) => fmt(t.tokens_per_s || 0) },
            { label: "Params", key: "n_params", num: true, render: (t) => fmt(t.n_params || 0) },
            { label: "Paper claim", key: "claim", render: (t) => `<span class="claim">${esc(t.meta?.claim || "")}</span>` },
            { label: "Source", key: "papers", render: (t) => (t.meta?.paper_records || [])
              .map((p) => `<a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.authors)} ${p.year}</a>`)
              .join("<br>") },
          ], rt)}
        </div>`;
    }).join("")}

    ${sum.champion_config ? `<div class="section-title">Champion configuration</div>
      <div class="grid cols-2">
        ${card("Discovered recipe", `Written to <code>${esc(sum.champion_yaml || "")}</code>.`,
          `<pre>${esc(JSON.stringify(sum.champion_config.model, null, 1))}</pre>`)}
        ${card("Seed-level validation",
          "Champion and baseline retrained across fresh seeds, to confirm the climb did not just fit the search.",
          `<dl class="kv">
            <dt>baseline seeds</dt><dd>${(sum.baseline_seed_losses || []).map((v) => num(v, 4)).join(", ")}</dd>
            <dt>champion seeds</dt><dd>${(sum.champion_seed_losses || []).map((v) => num(v, 4)).join(", ")}</dd>
            <dt>mean improvement</dt><dd class="neg">${num(sum.improvement, 4)} nats</dd>
            <dt>Cohen's d</dt><dd>${num(sum.effect_size_cohens_d, 2)}</dd>
          </dl>`)}
      </div>` : ""}`;

  $("#st-sel").addEventListener("change", (e) => { state.study = e.target.value; render("autoresearch"); });

  CH.line($("#c-traj"), {
    series: [{ name: "incumbent val loss", points: traj.filter((p) => Number.isFinite(p[1])) }],
    xLabel: "accepted moves", yLabel: "val loss (nats)", height: 230,
    band: Number.isFinite(sum.baseline_mean)
      ? { y0: sum.baseline_mean - sigma, y1: sum.baseline_mean + sigma, label: "baseline ±1σ" } : null,
    xFmt: (v) => (Number.isInteger(v) ? String(v) : ""),
    tipFmt: (v) => num(v, 4),
  });

  for (const rn of rounds) {
    const rt = cand.filter((t) => t.round === rn && Number.isFinite(t.delta))
      .sort((a, b) => a.delta - b.delta);
    CH.bars($(`#c-r${rn}`), {
      horizontal: true, rowH: 24, labelW: 150, diverging: true,
      height: Math.max(120, rt.length * 24 + 52),
      band: { lo: -thresh, hi: thresh, label: "within noise" },
      xLabel: "Δ validation loss (nats) — negative is better",
      items: rt.map((t) => ({
        label: t.intervention,
        value: t.delta,
        color: t.accepted ? CH.palette().good
          : (t.delta < -thresh ? CH.palette().series[0]
            : (t.delta > thresh ? CH.palette().critical : CH.palette().muted)),
        tip: `<div class="tt-title">${esc(t.meta?.title || t.intervention)}</div>
              <div class="tt-row"><span>Δ loss</span><b>${num(t.delta, 4)}</b></div>
              <div class="tt-row"><span>z-score</span><b>${num(t.z_score, 2)}</b></div>
              <div class="tt-row"><span>val loss</span><b>${num(t.val_loss, 4)}</b></div>
              <div class="tt-row"><span>tok/s</span><b>${fmt(t.tokens_per_s || 0)}</b></div>
              <div style="margin-top:5px;font-size:11.5px;color:var(--text-2)">${esc(t.meta?.expect || "")}</div>`,
      })),
      valueFmt: (v) => (v > 0 ? "+" : "") + v.toFixed(3),
    });
  }
}
const mean = (a) => a.reduce((x, y) => x + y, 0) / (a.length || 1);
const stdev = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2)) * a.length / Math.max(a.length - 1, 1)); };

/* ------------------------------------------------------------------ papers */
async function vPapers(c) {
  const d = await api("/api/papers");
  const cats = [...new Set(d.interventions.map((i) => i.category))].sort();
  c.innerHTML = `
    <div class="grid cols-4">
      ${stat("Papers", d.papers.length, "arXiv metadata fetched programmatically")}
      ${stat("Interventions", d.interventions.length, "config-level, independently ablatable")}
      ${stat("Categories", cats.length, cats.join(" · "))}
      ${stat("Years covered", `${Math.min(...d.papers.map((p) => p.year))}–${Math.max(...d.papers.map((p) => p.year))}`, "foundational to current")}
    </div>

    <div class="section-title">Interventions and the claims they test</div>
    <div class="card">${table([
      { label: "Key", key: "key", render: (i) => `<code>${esc(i.key)}</code>` },
      { label: "Technique", key: "title", render: (i) => `<b>${esc(i.title)}</b>` },
      { label: "Category", key: "category", render: (i) => `<span class="pill pill-muted">${esc(i.category)}</span>` },
      { label: "Claim under test", key: "claim", render: (i) => `<span class="claim">${esc(i.claim)}</span>` },
      { label: "Predicted signal", key: "expect", render: (i) => `<span class="claim">${esc(i.expect)}</span>` },
      { label: "Config change", key: "on", render: (i) =>
        `<code>${esc(Object.entries(i.on).map(([k, v]) => `${k.split(".").pop()}=${v}`).join(", "))}</code>` },
      { label: "Papers", key: "papers", render: (i) => i.paper_records
        .map((p) => `<a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.authors)} ${p.year}</a>`).join("<br>") },
    ], d.interventions)}</div>

    <div class="section-title">Full registry</div>
    <div class="controls">
      <input type="text" id="paper-q" placeholder="filter by title, author, tag…" style="min-width:280px">
      <span class="dim" id="paper-count"></span>
    </div>
    <div class="card"><div id="paper-tbl"></div></div>`;

  const draw = (q = "") => {
    const rows = d.papers.filter((p) => !q ||
      (p.title + p.authors + (p.tags || []).join(" ") + (p.arxiv || "")).toLowerCase().includes(q.toLowerCase()));
    $("#paper-count").textContent = `${rows.length} of ${d.papers.length}`;
    $("#paper-tbl").innerHTML = table([
      { label: "Year", key: "year", num: true },
      { label: "Authors", key: "authors" },
      { label: "Title", key: "title", render: (p) =>
        `<a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a>` },
      { label: "Ref", key: "arxiv", render: (p) => p.arxiv ? `<code>arXiv:${esc(p.arxiv)}</code>` : `<span class="dim">non-arXiv</span>` },
      { label: "Relevance", key: "claim", render: (p) => `<span class="claim">${esc(p.claim)}</span>` },
      { label: "Tested as", key: "used_by", render: (p) => (p.used_by || [])
        .map((u) => `<code>${esc(u)}</code>`).join(" ") || `<span class="dim">context</span>` },
    ], rows);
  };
  draw();
  $("#paper-q").addEventListener("input", (e) => draw(e.target.value));
}

/* ------------------------------------------------------------- model card */
async function vModel(c) {
  const m = await api("/api/model-card");
  const cfg = m.config?.model || {}, tr = m.training_summary || {}, ev = m.evaluation || {};
  const pb = m.param_breakdown || {};
  const total = Object.values(pb).reduce((a, b) => a + b, 0) || 1;

  c.innerHTML = `
    <div class="grid cols-4">
      ${stat("Parameters", fmt(m.params_total), `${fmt(m.params_non_embedding)} non-embedding`)}
      ${stat("Architecture", `${cfg.n_layer}L · ${cfg.d_model}d`, `${cfg.n_head} heads / ${cfg.n_kv_head} KV · ctx ${cfg.max_seq_len}`)}
      ${stat("Decode speed", num(ev.latency?.decode_tok_per_s, 1), `on ${esc(m.device)}`, "tok/s")}
      ${stat("KV cache", num(m.kv_cache_mb_full_ctx, 2), "full context, batch 1", "MB")}
    </div>

    <div class="section-title">Architecture</div>
    <div class="grid cols-2">
      ${card("Primitives in this checkpoint", "Every row is a config flag, and every flag traces to a paper in the registry.",
        `<dl class="kv">
          <dt>normalization</dt><dd>${esc(cfg.norm)} (${esc(cfg.norm_placement)}-norm)</dd>
          <dt>position</dt><dd>${esc(cfg.pos)}${cfg.pos === "rope" ? ` (θ=${cfg.rope_theta})` : ""}</dd>
          <dt>feed-forward</dt><dd>${esc(cfg.ffn)} (d_ff ${cfg.d_ff})</dd>
          <dt>attention</dt><dd>${cfg.n_head === cfg.n_kv_head ? "multi-head" : `grouped-query (${cfg.n_head / cfg.n_kv_head}× sharing)`}</dd>
          <dt>qk-norm</dt><dd>${cfg.qk_norm}</dd>
          <dt>tied embeddings</dt><dd>${cfg.tie_embeddings}</dd>
          <dt>z-loss</dt><dd>${cfg.z_loss}</dd>
          <dt>logit soft-cap</dt><dd>${cfg.logit_soft_cap || "off"}</dd>
          <dt>dropout</dt><dd>${cfg.dropout}</dd>
          <dt>vocabulary</dt><dd>${m.vocab_size} (byte-level BPE)</dd>
        </dl>`)}
      ${card("Parameter budget", "Where the capacity actually sits. Tied embeddings are counted once.",
        chartBox("c-params", 210))}
    </div>

    <div class="section-title">Intended use & limitations</div>
    <div class="grid cols-2">
      ${card("Intended use",
        "Following the Model Cards framework (Mitchell et al. 2018, arXiv:1810.03993).",
        `<p class="claim">A ${fmt(m.params_total)}-parameter demonstration model trained on synthetic
        children's stories (TinyStories, arXiv:2305.07759). It exists to make the full CRISP-DM cycle —
        data profiling through deployment — inspectable end to end on a single laptop GPU. It answers
        requests to write short, simple stories, and nothing else.</p>`)}
      ${card("Limitations", "Stated plainly, because the scale makes them severe.",
        `<ul class="claim" style="margin:0;padding-left:17px;display:flex;flex-direction:column;gap:5px">
          <li><b>Domain-locked.</b> The corpus is ~1500 words of simple English about children's
            situations. Anything outside that distribution produces fluent nonsense.</li>
          <li><b>No factual grounding.</b> There is no world knowledge in ${fmt(m.params_total)}
            parameters trained on ${fmt(m.training_summary?.tokens_seen || 0)} tokens. It should never be
            asked a question of fact.</li>
          <li><b>Context ${cfg.max_seq_len} tokens.</b> Longer conversations are truncated from the left.</li>
          <li><b>No safety tuning.</b> Supervised fine-tuning only — no preference optimization, no
            refusal training, no red-teaming.</li>
          <li><b>Under-trained relative to its size</b> (${num(tr.chinchilla_ratio, 2)}× the
            compute-optimal token budget), so quality is data-limited rather than capacity-limited.</li>
        </ul>`)}
    </div>

    <div class="section-title">Provenance</div>
    <div class="grid cols-2">
      ${card("Checkpoint & training", "",
        `<dl class="kv">
          <dt>checkpoint</dt><dd>${esc(m.checkpoint)}</dd>
          <dt>run id</dt><dd>${esc(m.run_id || "–")}</dd>
          <dt>step</dt><dd>${m.step ?? "–"}</dd>
          <dt>val loss</dt><dd>${num(m.val_loss, 4)}</dd>
          <dt>optimizer</dt><dd>${esc(m.config?.train?.optimizer || "")} · ${esc(m.config?.train?.schedule || "")}</dd>
          <dt>tokens seen</dt><dd>${fmt(tr.tokens_seen || 0)}</dd>
          <dt>train time</dt><dd>${dur(tr.elapsed_s)}</dd>
        </dl>`)}
      ${card("Tokenizer", "Trained from scratch on the training split only.",
        `<dl class="kv">
          <dt>algorithm</dt><dd>byte-level BPE</dd>
          <dt>vocabulary</dt><dd>${m.tokenizer_stats?.vocab_size ?? "–"}</dd>
          <dt>merges</dt><dd>${m.tokenizer_stats?.n_merges ?? "–"}</dd>
          <dt>compression</dt><dd>${num(m.tokenizer_stats?.bytes_per_token, 3)} bytes/token</dd>
          <dt>trained on</dt><dd>${fmt(m.tokenizer_stats?.n_chars || 0)} chars</dd>
          <dt>train time</dt><dd>${num(m.tokenizer_stats?.train_seconds, 1)} s</dd>
        </dl>`)}
    </div>`;

  CH.bars($("#c-params"), {
    horizontal: true, rowH: 24, labelW: 132, height: Math.max(120, Object.keys(pb).length * 24 + 48),
    items: Object.entries(pb).map(([k, v]) => ({
      label: k.replace(/_/g, " "), value: v,
      tip: `<div class="tt-title">${esc(k)}</div>
            <div class="tt-row"><span>parameters</span><b>${v.toLocaleString()}</b></div>
            <div class="tt-row"><span>share</span><b>${pct(v / total, 1)}</b></div>` })),
    valueFmt: (v) => fmt(v), xLabel: "parameters",
  });
}

/* -------------------------------------------------------------------- chat */
async function vChat(c) {
  let cks = [];
  try { cks = await api("/api/checkpoints"); } catch { /* none yet */ }
  const sft = cks.filter((k) => k.phase === "sft");
  c.innerHTML = `
    <div class="chat-wrap">
      <div class="card">
        <h3>Chat</h3>
        <p class="note">${sft.length ? "Served from the instruction-tuned checkpoint."
          : "No instruction-tuned checkpoint found — this is a raw pretrained model and will continue text rather than answer."}</p>
        <div class="chat-log" id="log"></div>
        <div class="chat-input">
          <textarea id="msg" placeholder="Write a short story that uses the words: cat, hat, run."
                    rows="2"></textarea>
          <button class="btn" id="send">Send</button>
        </div>
        <div class="gen-stats" id="gstats"></div>
      </div>
      <div class="card">
        <h3>Decoding</h3>
        <p class="note">These are the same knobs the evaluation sweep varies.</p>
        <div class="slider-row"><label>temperature <b id="v-temp">0.80</b></label>
          <input type="range" id="s-temp" min="0" max="1.5" step="0.05" value="0.8"></div>
        <div class="slider-row"><label>top-p <b id="v-topp">0.90</b></label>
          <input type="range" id="s-topp" min="0.1" max="1" step="0.05" value="0.9"></div>
        <div class="slider-row"><label>repetition penalty <b id="v-rp">1.05</b></label>
          <input type="range" id="s-rp" min="1" max="1.5" step="0.01" value="1.05"></div>
        <div class="slider-row"><label>max new tokens <b id="v-max">220</b></label>
          <input type="range" id="s-max" min="40" max="480" step="20" value="220"></div>
        <hr class="hairline">
        <label style="font-size:12px;color:var(--text-2)">Checkpoint
          <select id="ck" style="width:100%;margin-top:4px">
            ${cks.map((k) => `<option value="${esc(k.path)}" ${k.phase === "sft" ? "selected" : ""}>
              ${esc(k.phase)} · ${esc(k.run_id)} · ${num(k.best_val_loss, 3)}</option>`).join("")}
          </select></label>
        <button class="ghost-btn" id="reset" style="margin-top:11px;width:100%">Clear conversation</button>
      </div>
    </div>`;

  const log = $("#log");
  const paint = () => {
    log.innerHTML = state.chat.map((m) => `
      <div class="msg ${m.role === "user" ? "user" : "bot"}">
        <div class="who">${m.role === "user" ? "you" : "model"}</div>
        <div class="body">${esc(m.content)}</div></div>`).join("")
      || `<div class="empty" style="border:0">Ask for a story to begin.</div>`;
    log.scrollTop = log.scrollHeight;
  };
  paint();

  for (const [s, v, d] of [["s-temp", "v-temp", 2], ["s-topp", "v-topp", 2],
                           ["s-rp", "v-rp", 2], ["s-max", "v-max", 0]]) {
    $(`#${s}`).addEventListener("input", (e) => { $(`#${v}`).textContent = (+e.target.value).toFixed(d); });
  }
  $("#reset").addEventListener("click", () => { state.chat = []; paint(); $("#gstats").textContent = ""; });

  async function send() {
    const box = $("#msg");
    const text = box.value.trim();
    if (!text || state.busy) return;
    state.busy = true; $("#send").disabled = true;
    box.value = "";
    state.chat.push({ role: "user", content: text });
    state.chat.push({ role: "assistant", content: "" });
    paint();

    const body = JSON.stringify({
      messages: state.chat.slice(0, -1),
      temperature: +$("#s-temp").value, top_p: +$("#s-topp").value,
      repetition_penalty: +$("#s-rp").value, max_new_tokens: +$("#s-max").value,
      checkpoint: $("#ck").value || null,
    });
    try {
      const r = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" }, body });
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop();
        for (const p of parts) {
          if (!p.startsWith("data: ")) continue;
          const ev = JSON.parse(p.slice(6));
          if (ev.error) { state.chat.at(-1).content += `\n[error: ${ev.error}]`; }
          else if (ev.token) { state.chat.at(-1).content += ev.token; }
          else if (ev.done) {
            const s = ev.stats;
            $("#gstats").textContent =
              `${s.completion_tokens} tokens · ${s.decode_tok_per_s} tok/s · TTFT ${s.ttft_ms} ms · KV ${s.kv_cache_mb} MB`;
          }
          paint();
        }
      }
    } catch (e) {
      state.chat.at(-1).content += `\n[request failed: ${e.message}]`;
      paint();
    }
    state.busy = false; $("#send").disabled = false;
    $("#msg").focus();
  }
  $("#send").addEventListener("click", send);
  $("#msg").addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.keyCode === 13) && !e.shiftKey) { e.preventDefault(); send(); }
  });
}

/* --------------------------------------------------------------- bootstrap */
$$(".nav-item").forEach((b) => b.addEventListener("click", () => render(b.dataset.view)));
$("#refresh").addEventListener("click", () => render(state.view));
$("#theme-toggle").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const next = cur === "dark" ? "light" : cur === "light" ? "dark"
    : (matchMedia("(prefers-color-scheme: dark)").matches ? "light" : "dark");
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("slm-theme", next); } catch { /* private mode */ }
  CH.rerenderAll();
});
try {
  const t = localStorage.getItem("slm-theme");
  if (t) document.documentElement.setAttribute("data-theme", t);
} catch { /* private mode */ }

api("/api/health").then((h) => {
  const el = $("#health");
  el.className = "pill " + (h.checkpoint ? "pill-good" : "pill-warn");
  el.textContent = h.checkpoint ? "model ready" : "no checkpoint yet";
}).catch(() => { $("#health").className = "pill pill-bad"; $("#health").textContent = "api offline"; });

render("overview");
