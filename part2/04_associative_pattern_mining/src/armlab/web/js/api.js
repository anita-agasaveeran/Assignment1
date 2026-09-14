/* Artifact access. Every panel reads from the FastAPI service so the dashboard
   reflects the artifacts on disk, not a snapshot baked into the page. */
const cache = new Map();

export async function get(path) {
  if (cache.has(path)) return cache.get(path);
  const p = fetch(path).then(async (r) => {
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).error || msg; } catch {}
      throw new Error(`${path}: ${msg}`);
    }
    return r.json();
  });
  cache.set(path, p);
  return p;
}
export const bust = () => cache.clear();
export async function post(path, body) {
  const r = await fetch(path, {method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify(body)});
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}
export const A = {
  run: () => get("/api/run"),
  profile: () => get("/api/profile"),
  preparation: () => get("/api/preparation"),
  itemsets: () => get("/api/itemsets"),
  autoresearch: () => get("/api/autoresearch"),
  evaluation: () => get("/api/evaluation"),
  properties: () => get("/api/properties"),
  network: () => get("/api/network"),
  modelCard: () => get("/api/model-card"),
  manifest: () => get("/api/manifest"),
  rules: (q = {}) => get("/api/rules?" + new URLSearchParams(q)),
  recommend: (body) => post("/api/recommend", body),
};
