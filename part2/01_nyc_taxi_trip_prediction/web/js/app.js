/* Application shell: wires the form, the map and the three insight panels to
 * the API.  Keeps no state of its own beyond the last sweep — the map owns the
 * endpoints, the server owns everything else. */

(() => {
  const $ = (sel) => document.querySelector(sel);
  const api = async (path, body) => {
    const res = await fetch(path, body ? {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    } : undefined);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail?.[0]?.msg || data.detail || `HTTP ${res.status}`);
    return data;
  };

  const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  let stats = null, places = [], lastSweep = null;

  /* ------------------------------------------------------------ helpers */
  const fmtCoord = (p) => p ? `${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}` : "not set";

  function localDatetimeValue(d = new Date()) {
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  function tripRequest() {
    const { pickup, dropoff } = TaxiMap.state;
    if (!pickup || !dropoff) throw new Error("Set both a pickup and a dropoff point first.");
    return {
      pickup_latitude: pickup.lat, pickup_longitude: pickup.lon,
      dropoff_latitude: dropoff.lat, dropoff_longitude: dropoff.lon,
      pickup_datetime: $("#when").value || null,
      passenger_count: Number($("#pax").value),
      vendor_id: 1,
    };
  }

  /* The result card describes a specific route at a specific time.  When
   * either changes, the card is stale until re-estimated. */
  function markStale(on) {
    const card = $("#result");
    if (card.hidden) return;
    card.classList.toggle("is-stale", on);
    $("#res-stale").hidden = !on;
  }

  function showError(msg) {
    const box = $("#form-error");
    box.textContent = msg;
    box.hidden = !msg;
  }

  /* ------------------------------------------------------------- health */
  async function loadHealth() {
    const pill = $("#health");
    try {
      const h = await api("/api/health");
      const ok = h.model_loaded;
      pill.className = `pill ${ok ? "pill-ok" : "pill-bad"}`;
      pill.lastElementChild.textContent = ok ? "model live" : "model missing";
      if (!ok) showError(h.error || "The model bundle is not loaded. Run `python -m taxi.train`.");
    } catch {
      pill.className = "pill pill-bad";
      pill.lastElementChild.textContent = "api offline";
    }
  }

  /* ------------------------------------------------------------ estimate */
  async function estimate() {
    showError("");
    let req;
    try { req = tripRequest(); } catch (e) { return showError(e.message); }

    const btn = $("#estimate");
    btn.disabled = true;
    btn.textContent = "Estimating…";
    try {
      // One round trip for the answer, one for the hour sweep behind it.
      const [pred, sweep] = await Promise.all([
        api("/api/predict", req),
        api("/api/predict/by-hour", req),
      ]);
      renderResult(pred);
      renderSweep(sweep, new Date(pred.pickup_datetime).getHours());
      TaxiMap.fit();
    } catch (e) {
      showError(e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Estimate duration";
    }
  }

  function renderResult(p) {
    $("#result").hidden = false;
    markStale(false);
    $("#res-min").textContent = p.duration_minutes.toFixed(1);
    $("#res-range").textContent =
      `${p.interval_pct}% of trips like this take ${(p.low_seconds / 60).toFixed(0)}–${(p.high_seconds / 60).toFixed(0)} min`;
    $("#res-eta").textContent = p.eta;
    $("#res-grid").textContent = p.grid_km.toFixed(1);
    $("#res-straight").textContent = p.straight_line_km.toFixed(1);
    $("#res-speed").textContent = p.avg_speed_kmh ?? "–";
    $("#res-notes").replaceChildren(
      ...p.notes.map((n) => Object.assign(document.createElement("li"), { textContent: n }))
    );
  }

  function renderSweep(sweep, selectedHour) {
    lastSweep = sweep;
    Charts.sweep($("#sweep-chart"), sweep.series, { selectedHour });
    const box = $("#sweep-summary");
    const fmtHour = (h) => `${String(h).padStart(2, "0")}:00`;
    box.innerHTML =
      `Leaving at <b>${fmtHour(sweep.best_hour)}</b> is the fastest option on ` +
      `${sweep.date}; <b>${fmtHour(sweep.worst_hour)}</b> is the worst, and the ` +
      `gap between them is <b>${sweep.spread_minutes} minutes</b> on the same route.`;
    box.hidden = false;
  }

  /* ---------------------------------------------------------------- data */
  function renderData() {
    if (!stats) return;
    const d = stats.duration, s = stats.speed, dist = stats.distance;
    const kpis = [
      ["trips analysed", stats.n_trips.toLocaleString()],
      ["median trip", `${(d.median_s / 60).toFixed(1)} min`],
      ["median distance", `${dist.median_grid_km.toFixed(1)} km`],
      ["p90 trip", `${(d.p90_s / 60).toFixed(0)} min`],
      ["PM-rush speed", `${s.rush_pm_kmh.toFixed(1)} km/h`],
      ["overnight speed", `${s.overnight_kmh.toFixed(1)} km/h`],
    ];
    $("#data-kpis").replaceChildren(...kpis.map(([k, v]) => {
      const el = document.createElement("div");
      el.className = "kpi";
      el.innerHTML = `<div class="v">${v}</div><div class="k">${k}</div>`;
      return el;
    }));

    const h = stats.duration_hist;
    Charts.columns($("#duration-chart"), h.counts, {
      color: Charts.COLORS.blue,
      fmt: (v) => v >= 1000 ? `${Math.round(v / 1000)}k` : Math.round(v),
      labels: [[0, "0"], [Math.floor(h.counts.length / 2), "30m"], [h.counts.length - 1, "60m"]],
    });

    Charts.heatmap($("#heat-chart"), stats.heatmap_dow_hour, {
      rows: DOW,
      cols: [[0, "0"], [6, "6"], [12, "12"], [18, "18"], [23, "23"]],
    });

    const audit = stats.cleaning_audit || [];
    if (audit.length) {
      const total = audit.find((a) => a.rule === "TOTAL");
      const rows = audit.filter((a) => a.rule !== "TOTAL");
      const start = rows[0].dropped + rows[0].remaining;
      $("#audit-table").innerHTML = `
        <table>
          <thead><tr><th>rule</th><th>dropped</th><th>%</th></tr></thead>
          <tbody>
            ${rows.map((a) => `<tr><td>${a.rule}</td><td>${a.dropped.toLocaleString()}</td>
              <td>${(100 * a.dropped / start).toFixed(2)}%</td></tr>`).join("")}
            <tr class="best"><td>kept</td><td>${total.remaining.toLocaleString()}</td>
              <td>${(100 * total.remaining / start).toFixed(1)}%</td></tr>
          </tbody>
        </table>`;
    }

    const byHour = stats.by_hour;
    Charts.line($("#speed-chart"), byHour.map((r) => r.median_speed), {
      color: Charts.COLORS.green, unit: "km/h", fmt: (v) => v.toFixed(0),
      labels: [[0, "0:00"], [6, "6:00"], [12, "12:00"], [18, "18:00"], [23, "23:00"]],
    });
  }

  /* --------------------------------------------------------------- model */
  async function loadModel() {
    let m;
    try { m = await api("/api/metrics"); } catch (e) {
      $("#metrics-table").innerHTML = `<div class="empty">${e.message}</div>`;
      return;
    }
    const rows = m.metrics.results;
    // The winner is whichever model has the lowest RMSLE — the Kaggle metric.
    const bestRmsle = Math.min(...rows.map((r) => r.rmsle));

    $("#metrics-table").innerHTML = `
      <table>
        <thead><tr><th>model</th><th>RMSLE</th><th>MAE</th><th>±5 min</th></tr></thead>
        <tbody>
          ${rows.map((r) => `
            <tr class="${r.rmsle === bestRmsle ? "best" : ""}">
              <td>${r.model.replace(/_/g, " ")}</td>
              <td>${r.rmsle.toFixed(4)}</td>
              <td>${(r.mae_seconds / 60).toFixed(1)} min</td>
              <td>${r.within_5min_pct.toFixed(0)}%</td>
            </tr>`).join("")}
        </tbody>
      </table>
      <p class="hint" style="margin-top:12px">
        ${m.metrics.n_train.toLocaleString()} training trips through
        ${m.metrics.train_period[1].slice(0, 10)}; scored on
        ${m.metrics.n_test.toLocaleString()} later trips.
      </p>`;

    Charts.ranked($("#importance-chart"), m.feature_importance.map((f) => ({
      label: f.feature, value: f.importance, display: f.importance.toFixed(3),
    })));

    const iv = m.metrics.interval;
    $("#interval-note").innerHTML =
      `The p${iv.quantiles[0] * 100}–p${iv.quantiles[1] * 100} band is meant to cover ` +
      `<b>${iv.nominal_coverage_pct}%</b> of trips; on held-out data it actually covers ` +
      `<b>${iv.empirical_coverage_pct.toFixed(1)}%</b>. Close to nominal, and slightly ` +
      `narrow — the band is a touch optimistic on the hardest trips.`;
  }

  /* ---------------------------------------------------------------- init */
  async function init() {
    TaxiMap.init("map");
    TaxiMap.onChange((s, meta = {}) => {
      $("#pickup-coords").textContent = fmtCoord(s.pickup);
      $("#pickup-coords").classList.toggle("unset", !s.pickup);
      $("#dropoff-coords").textContent = fmtCoord(s.dropoff);
      $("#dropoff-coords").classList.toggle("unset", !s.dropoff);

      // A point moved by hand is no longer the landmark the select claims, and
      // the estimate on screen no longer describes the route on the map.  Say
      // so rather than leaving stale numbers looking authoritative.
      if (meta.role && meta.source !== "landmark") {
        const sel = $(`#${meta.role}-select`);
        if (sel) sel.value = "";
      }
      if (meta.source === "clear") markStale(false);
      else if (meta.source) markStale(true);

      $("#click-hint").innerHTML = !s.pickup
        ? "Click the map to set the <b>pickup</b>, then click again for the <b>dropoff</b>."
        : !s.dropoff
          ? "Now click the <b>dropoff</b>. Drag either marker to fine-tune."
          : "Drag a marker or change the departure time, then re-estimate.";
    });

    $("#when").value = localDatetimeValue(new Date("2016-06-15T18:00:00"));
    $("#estimate").addEventListener("click", estimate);
    $("#when").addEventListener("change", () => markStale(true));
    $("#pax").addEventListener("change", () => markStale(true));
    $("#clear").addEventListener("click", () => {
      TaxiMap.clear(); $("#result").hidden = true; showError("");
    });
    $("#swap").addEventListener("click", () => { TaxiMap.swap(); estimate(); });

    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t === tab));
        document.querySelectorAll(".tabpanel").forEach((p) =>
          p.classList.toggle("is-active", p.dataset.panel === tab.dataset.tab));
      });
    });

    $("#layer-density").addEventListener("change", (e) =>
      TaxiMap.toggleDensity(stats?.pickup_points, e.target.checked));
    $("#layer-zones").addEventListener("change", async (e) => {
      if (!e.target.checked) return TaxiMap.toggleZones([], false);
      const { clusters } = await api("/api/zones");
      TaxiMap.toggleZones(clusters, true);
    });

    loadHealth();
    loadModel();

    // Landmark pickers: a fast path for anyone who does not want to hunt for
    // JFK on a dark basemap.
    try {
      const res = await api("/api/places");
      places = res.places;
      ["pickup", "dropoff"].forEach((role) => {
        const sel = $(`#${role}-select`);
        places.forEach((p, i) => sel.append(new Option(p.name, String(i))));
        sel.addEventListener("change", () => {
          const p = places[Number(sel.value)];
          if (p) TaxiMap.setPoint(role, p.lat, p.lon, { pan: true, source: "landmark" });
        });
      });
    } catch { /* the map still works without the shortcut list */ }

    try {
      stats = await api("/api/stats");
      renderData();
    } catch {
      $("#data-kpis").innerHTML = `<div class="empty">Run \`python -m taxi.eda\` to populate this tab.</div>`;
    }

    // Seed a real trip so the page is never an empty canvas: Times Square to
    // JFK at 6pm, the case where the model has the most to say.  Seeding
    // *through the landmark selects* keeps the form and the map in agreement.
    seedFrom("Times Square", "JFK Airport");
    TaxiMap.fit();
    estimate();
  }

  function seedFrom(pickupName, dropoffName) {
    const pick = (role, name) => {
      const i = places.findIndex((p) => p.name === name);
      if (i < 0) return false;
      const sel = document.querySelector(`#${role}-select`);
      if (sel) sel.value = String(i);
      TaxiMap.setPoint(role, places[i].lat, places[i].lon, { source: "landmark" });
      return true;
    };
    if (!pick("pickup", pickupName) || !pick("dropoff", dropoffName)) {
      // The landmark list failed to load; fall back to raw coordinates.
      TaxiMap.setPoint("pickup", 40.7580, -73.9855);
      TaxiMap.setPoint("dropoff", 40.6413, -73.7781);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
