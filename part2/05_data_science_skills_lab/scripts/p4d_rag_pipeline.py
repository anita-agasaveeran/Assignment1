"""CRISP-DM Phase 4 - Modeling (Track A: retrieval-augmented generation).

Skill demonstrated:
  * rag-pipeline -- ingest & chunk, embed, dense + sparse (BM25) retrieval,
                    cross-encoder reranking, prompt assembly with citations,
                    and a measured retrieval evaluation (recall@k, MRR).

The corpus is this project's own analysis output: the reports, schema map and
metric definitions produced in Phases 2-4. That makes the evaluation honest --
the answers genuinely live in the corpus and can be checked.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

from common import REP, ROOT, TAB, banner, seed_everything, write_json, write_report

TOP_K = 20
FINAL_K = 5


# ------------------------------------------------------------- 1. ingest & chunk
def chunk_markdown(path: Path, target_words: int = 160, overlap: float = 0.15) -> list[dict]:
    """Structural chunking on headings, then size-capped with overlap.

    The skill prefers semantic/structural boundaries over fixed character counts,
    and requires metadata on every chunk so answers can carry citations.
    """
    text = path.read_text()
    sections, title, buf = [], "(preamble)", []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                sections.append((title, "\n".join(buf)))
            title, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((title, "\n".join(buf)))

    chunks = []
    for sec_title, body in sections:
        words = body.split()
        if not words:
            continue
        step = max(1, int(target_words * (1 - overlap)))
        for i in range(0, len(words), step):
            piece = " ".join(words[i:i + target_words])
            if len(piece.split()) < 20:
                continue
            chunks.append({
                "id": f"{path.stem}#{sec_title}#{i // step}",
                "text": f"{sec_title}. {piece}",
                "source": (f"{path.parent.name}/{path.name}" if path.name == "SKILL.md" else path.name),
                "section": sec_title,
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            })
            if i + target_words >= len(words):
                break
    return chunks


# ------------------------------------------------------------------ 3. retrieve
class BM25:
    """Sparse keyword retrieval. The skill's #1 pitfall is dense-only retrieval,
    which misses exact identifiers, codes and numbers."""

    def __init__(self, docs: list[str]):
        from rank_bm25 import BM25Okapi

        self.tok = [self._t(d) for d in docs]
        self.model = BM25Okapi(self.tok)

    @staticmethod
    def _t(s: str) -> list[str]:
        return re.findall(r"[a-z0-9_.%-]+", s.lower())

    def search(self, q: str, k: int) -> list[tuple[int, float]]:
        sc = self.model.get_scores(self._t(q))
        idx = np.argsort(sc)[::-1][:k]
        return [(int(i), float(sc[i])) for i in idx]


def rrf(*rankings: list[int], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: merge dense and sparse without score normalisation."""
    scores: dict[int, float] = {}
    for r in rankings:
        for rank, doc in enumerate(r):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]


# ------------------------------------------------------------------ 6. evaluate
def evaluate(name: str, results: dict[str, list[int]], gold: dict[str, set[int]],
             ks=(1, 3, 5, 10)) -> dict:
    out = {"retriever": name}
    for k in ks:
        hits = [len(gold[q] & set(results[q][:k])) > 0 for q in gold]
        out[f"recall@{k}"] = round(float(np.mean(hits)), 4)
    rr = []
    for q in gold:
        r = next((i + 1 for i, d in enumerate(results[q]) if d in gold[q]), None)
        rr.append(1.0 / r if r else 0.0)
    out["mrr"] = round(float(np.mean(rr)), 4)
    return out


def main() -> None:
    seed_everything()
    banner("rag-pipeline", "Phase 4 - Modeling",
           "chunk -> embed -> hybrid retrieve -> rerank -> cite -> measure")

    # -- Stage 1: ingest & chunk -------------------------------------------
    # Corpus = this project's own analysis reports PLUS the 46 installed SKILL.md
    # files. A 26-chunk corpus saturates any retriever; several hundred chunks with
    # heavy vocabulary overlap is a realistic retrieval problem.
    skill_docs = sorted((Path.home() / ".claude" / "skills").glob("*/SKILL.md"))
    # The corpus is PINNED to an explicit list, not a glob over outputs/reports.
    # A glob makes the corpus depend on which phases have already run, so the same
    # code produces different retrieval scores on a second pass -- the eval set has
    # to be fixed or the numbers are not comparable. These are the Phase 2 reports,
    # all of which exist before this script runs in the pipeline order.
    PINNED = ["schema_map.md", "query_review.md", "query_business_logic.md",
              "metric_reconciliation.md", "catalog_invoice_lines.md", "catalog_customers.md"]
    reports = [REP / n for n in PINNED]
    missing = [r.name for r in reports if not r.exists()]
    if missing:
        raise SystemExit(f"pinned corpus files missing (run Phase 2 first): {missing}")
    docs = reports + skill_docs
    chunks: list[dict] = []
    for d in docs:
        chunks.extend(chunk_markdown(d))   # no silent except: a chunking failure is a bug
    texts = [c["text"] for c in chunks]
    print(f"  stage 1 ingest & chunk")
    print(f"          {len(docs)} source documents ({len(skill_docs)} skill docs + "
          f"{len(reports)} analysis reports) -> {len(chunks)} chunks")
    print(f"          target 160 words, 15% overlap, split on markdown headings")
    print(f"          mean chunk length {np.mean([len(t.split()) for t in texts]):.0f} words")
    print(f"          every chunk carries source/section metadata for citations")

    # -- Stage 2: embed -----------------------------------------------------
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"\n  stage 2 embed")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                          batch_size=64)
    print(f"          model all-MiniLM-L6-v2, dim {emb.shape[1]}, L2-normalised")
    print(f"          cosine similarity == dot product on normalised vectors")
    print(f"          SAME model used at index and query time (a core pitfall)")

    # -- Stage 3: retrieve --------------------------------------------------
    bm25 = BM25(texts)
    print(f"\n  stage 3 retrieve: dense (cosine) + sparse (BM25 Okapi), fused with RRF")

    # -- Eval set: questions whose answers genuinely live in the corpus -----
    queries = [
        ("What is the gross revenue variance between the raw feed and the mart?",
         ["metric_reconciliation"]),
        ("Why was the December 2011 revenue drop not a business event?",
         ["rca_", "root", "metric_reconciliation"]),
        ("Which StockCode column is the primary key of the products table?",
         ["schema_map"]),
        ("What percentage of customers are Champions and what revenue share do they hold?",
         ["segment", "insight", "executive"]),
        ("What is the anti-pattern with strftime on a filtered date column?",
         ["query_review"]),
        ("How is average_order_value defined in the semantic layer?",
         ["schema_map", "query_business_logic", "metric_reconciliation"]),
        ("What join path connects invoice_lines to customers?",
         ["schema_map"]),
        ("Are cancellations included in the revenue metric?",
         ["metric_reconciliation", "query_business_logic"]),
        ("What is the repeat purchase rate benchmark for e-commerce?",
         ["metrics", "insight", "executive"]),
        ("Which countries drove the November to December change?",
         ["rca_", "root"]),
    ]

    gold: dict[str, set[int]] = {}
    for q, pats in queries:
        ids = {i for i, c in enumerate(chunks)
               if any(p.lower() in c["source"].lower() for p in pats)}
        gold[q] = ids
    usable = [(q, p) for q, p in queries if gold[q]]
    print(f"          eval set: {len(usable)} questions with gold chunks in the corpus")

    dense_r, sparse_r, hybrid_r, rerank_r = {}, {}, {}, {}
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    for q, _ in usable:
        qv = embedder.encode([q], normalize_embeddings=True)[0]
        sims = emb @ qv
        d_idx = np.argsort(sims)[::-1][:TOP_K].tolist()
        s_idx = [i for i, _ in bm25.search(q, TOP_K)]
        h_idx = rrf(d_idx, s_idx)[:TOP_K]

        # -- Stage 4: rerank the fused candidates with a cross-encoder ------
        pairs = [(q, texts[i]) for i in h_idx]
        scores = reranker.predict(pairs, show_progress_bar=False)
        r_idx = [i for _, i in sorted(zip(scores, h_idx), key=lambda x: -x[0])]

        dense_r[q], sparse_r[q], hybrid_r[q], rerank_r[q] = d_idx, s_idx, h_idx, r_idx

    g = {q: gold[q] for q, _ in usable}
    table = [evaluate("dense only", dense_r, g),
             evaluate("sparse only (BM25)", sparse_r, g),
             evaluate("hybrid (RRF)", hybrid_r, g),
             evaluate("hybrid + cross-encoder rerank", rerank_r, g)]

    print(f"\n  stage 6 retrieval evaluation ({len(usable)} queries)")
    hdr = f"  {'retriever':<32}" + "".join(f"{k:>10}" for k in
                                           ["recall@1", "recall@3", "recall@5", "recall@10", "MRR"])
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in table:
        print(f"  {r['retriever']:<32}" +
              f"{r['recall@1']:>10.3f}{r['recall@3']:>10.3f}{r['recall@5']:>10.3f}"
              f"{r['recall@10']:>10.3f}{r['mrr']:>10.3f}")

    best = max(table, key=lambda r: r["mrr"])
    dense_only, sparse_only, hyb, rer = table
    print(f"\n  -> best by MRR: {best['retriever']} ({best['mrr']:.3f})")
    print(f"     dense only  {dense_only['mrr']:.3f}")
    print(f"     sparse only {sparse_only['mrr']:.3f}")
    print(f"     hybrid      {hyb['mrr']:.3f}   ({hyb['mrr']-dense_only['mrr']:+.3f} vs dense-only)")
    print(f"     + rerank    {rer['mrr']:.3f}   ({rer['mrr']-hyb['mrr']:+.3f} vs hybrid)")
    print()
    if hyb["mrr"] > dense_only["mrr"] and hyb["mrr"] > sparse_only["mrr"]:
        print(f"     Fusing dense with BM25 beats either alone -- the skill's core claim,")
        print(f"     measured rather than asserted. Dense retrieval alone misses the exact")
        print(f"     identifiers (StockCode, strftime, invoice_lines) that BM25 matches directly.")
    else:
        print(f"     On this corpus fusion did NOT beat the better single retriever.")
    if rer["mrr"] >= hyb["mrr"]:
        print(f"     The cross-encoder rerank adds a further {rer['mrr']-hyb['mrr']:+.3f} MRR.")
    else:
        print(f"     REPORTED AS MEASURED: the cross-encoder rerank *lost* "
              f"{hyb['mrr']-rer['mrr']:.3f} MRR here.")
        print(f"     ms-marco-MiniLM is trained on web-search passages; these chunks are")
        print(f"     terse technical prose with heavy vocabulary overlap, which is out of")
        print(f"     domain for it. The skill's advice to always rerank is sound in general,")
        print(f"     but this is exactly why it also insists on measuring retrieval: a")
        print(f"     reranker adopted on faith would have made this system worse.")

    # -- Stage 5: prompt assembly with citations ---------------------------
    demo_q = usable[0][0]
    top = rerank_r[demo_q][:FINAL_K]
    print(f"\n  stage 5 prompt assembly for: \"{demo_q}\"")
    print(f"          top {FINAL_K} reranked chunks:")
    for rank, i in enumerate(top, 1):
        c = chunks[i]
        mark = "GOLD" if i in g[demo_q] else "    "
        print(f"          {rank}. [{mark}] {c['source']} :: {c['section'][:44]}")

    ctx = "\n\n".join(
        f"[{n}] source={chunks[i]['source']} section=\"{chunks[i]['section']}\"\n{chunks[i]['text'][:400]}"
        for n, i in enumerate(top, 1))
    prompt = f"""You are answering questions about an internal data-analysis project.

Answer ONLY from the context below. If the answer is not in the context, say
"I don't know from the provided context." Cite every claim inline as [n].

CONTEXT
{ctx}

QUESTION
{demo_q}

ANSWER (with inline [n] citations):"""

    write_report(
        "# RAG prompt assembly example\n\n"
        "*Produced by the `rag-pipeline` skill. The context block below is the top-5 "
        "output of hybrid retrieval + cross-encoder reranking over this project's own "
        "analysis reports.*\n\n```text\n" + prompt + "\n```\n",
        "rag_prompt_example.md")

    print(f"\n          assembled prompt: {len(prompt.split())} words, {FINAL_K} cited chunks")
    print(f"          instruction pins the model to the context and requires [n] citations")

    write_json({"n_documents": len(docs), "n_chunks": len(chunks),
                "embedding_model": "all-MiniLM-L6-v2", "embedding_dim": int(emb.shape[1]),
                "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2",
                "n_eval_queries": len(usable), "top_k": TOP_K, "final_k": FINAL_K,
                "evaluation": table,
                "best_retriever": best["retriever"],
                "mrr_gain_over_dense_only": round(best["mrr"] - dense_only["mrr"], 4)},
               "p4d_rag_results.json")

    import matplotlib.pyplot as plt

    from common import FIG, PALETTE, style_plots

    style_plots()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ks = ["recall@1", "recall@3", "recall@5", "recall@10"]
    w = 0.2
    x = np.arange(len(ks))
    for i, r in enumerate(table):
        ax.bar(x + i * w, [r[k] for k in ks], w, label=r["retriever"], color=PALETTE[i])
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(ks)
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval strategies compared on 46 skill docs + 7 reports")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p4d_rag_eval.png")
    plt.close(fig)
    print("  -> wrote p4d_rag_eval.png")


if __name__ == "__main__":
    main()
