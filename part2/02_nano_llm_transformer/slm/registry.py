"""Paper registry + the AutoResearch intervention search space.

Every citation below was verified against the arXiv API (title / first author /
year fetched programmatically, not from memory). Non-arXiv works carry an
explicit `url` instead.

The point of this module: an AutoResearch trial is never "try knob X". It is
"test the claim made by paper P, on our data, at our scale, and record the
measured effect next to the claimed effect." The dashboard renders exactly that
join, so every bar in the AutoResearch view traces back to a real citation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# Papers. key -> metadata. `claim` is our one-line summary of what the paper
# asserts that is *testable at our scale*; it is the string the dashboard shows
# beside the measured delta.
# --------------------------------------------------------------------------
PAPERS: dict[str, dict[str, Any]] = {
    # --- foundations -------------------------------------------------------
    "vaswani2017": dict(
        arxiv="1706.03762", year=2017, authors="Vaswani et al.",
        title="Attention Is All You Need",
        claim="Self-attention alone, without recurrence or convolution, suffices for sequence transduction.",
        tags=["architecture", "foundation"]),
    "radford2019gpt2": dict(
        arxiv=None, year=2019, authors="Radford et al.",
        url="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
        title="Language Models are Unsupervised Multitask Learners (GPT-2)",
        claim="Decoder-only LM pretraining scales to multitask ability; scale residual-projection init by 1/sqrt(2L).",
        tags=["architecture", "initialization", "foundation"]),
    "brown2020gpt3": dict(
        arxiv="2005.14165", year=2020, authors="Brown et al.",
        title="Language Models are Few-Shot Learners",
        claim="Scale alone produces in-context few-shot learning.",
        tags=["scaling", "foundation"]),
    "xiong2020prenorm": dict(
        arxiv="2002.04745", year=2020, authors="Xiong et al.",
        title="On Layer Normalization in the Transformer Architecture",
        claim="Pre-LN keeps gradient magnitudes bounded at init, so training is stable without a warmup stage.",
        tags=["normalization", "stability"]),
    "he2015resnet": dict(
        arxiv="1512.03385", year=2015, authors="He et al.",
        title="Deep Residual Learning for Image Recognition",
        claim="Identity shortcuts make very deep networks optimizable.",
        tags=["architecture", "foundation"]),

    # --- normalization -----------------------------------------------------
    "ba2016layernorm": dict(
        arxiv="1607.06450", year=2016, authors="Ba et al.",
        title="Layer Normalization",
        claim="Per-sample feature normalization stabilizes training independent of batch size.",
        tags=["normalization"]),
    "zhang2019rmsnorm": dict(
        arxiv="1910.07467", year=2019, authors="Zhang et al.",
        title="Root Mean Square Layer Normalization",
        claim="Dropping mean-centering from LayerNorm matches quality at 7-64% lower normalization cost.",
        tags=["normalization", "efficiency"]),
    "henry2020qknorm": dict(
        arxiv="2010.04245", year=2020, authors="Henry et al.",
        title="Query-Key Normalization for Transformers",
        claim="L2-normalizing queries and keys bounds attention logits and improves low-resource NMT.",
        tags=["normalization", "attention", "stability"]),
    "dehghani2023vit22b": dict(
        arxiv="2302.05442", year=2023, authors="Dehghani et al.",
        title="Scaling Vision Transformers to 22 Billion Parameters",
        claim="QK-norm removes the attention-logit divergence that otherwise kills large-model training.",
        tags=["normalization", "stability", "scaling"]),
    "wortsman2023proxies": dict(
        arxiv="2309.14322", year=2023, authors="Wortsman et al.",
        title="Small-scale proxies for large-scale Transformer training instabilities",
        claim="Instabilities (logit growth, attention collapse) reproduce at small scale; qk-norm and z-loss fix them.",
        tags=["stability", "methodology", "autoresearch"]),

    # --- positional --------------------------------------------------------
    "su2021rope": dict(
        arxiv="2104.09864", year=2021, authors="Su et al.",
        title="RoFormer: Enhanced Transformer with Rotary Position Embedding",
        claim="Rotating q/k by position encodes relative offsets inside the dot product, with no learned position table.",
        tags=["positional"]),
    "press2021alibi": dict(
        arxiv="2108.12409", year=2021, authors="Press et al.",
        title="Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation",
        claim="A linear distance bias on attention logits extrapolates beyond the training context length.",
        tags=["positional", "long-context"]),
    "peng2023yarn": dict(
        arxiv="2309.00071", year=2023, authors="Peng et al.",
        title="YaRN: Efficient Context Window Extension of Large Language Models",
        claim="Rescaling RoPE frequencies extends usable context with little additional training.",
        tags=["positional", "long-context"]),
    "beltagy2020longformer": dict(
        arxiv="2004.05150", year=2020, authors="Beltagy et al.",
        title="Longformer: The Long-Document Transformer",
        claim="Sliding-window attention gives linear-time context at near-full-attention quality.",
        tags=["attention", "long-context"]),

    # --- attention ---------------------------------------------------------
    "shazeer2019mqa": dict(
        arxiv="1911.02150", year=2019, authors="Shazeer",
        title="Fast Transformer Decoding: One Write-Head is All You Need",
        claim="Sharing a single K/V head collapses the decoding memory bottleneck at a small quality cost.",
        tags=["attention", "inference"]),
    "ainslie2023gqa": dict(
        arxiv="2305.13245", year=2023, authors="Ainslie et al.",
        title="GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
        claim="Grouped K/V heads recover multi-head quality while keeping most of multi-query's speedup.",
        tags=["attention", "inference", "efficiency"]),
    "dao2022flashattention": dict(
        arxiv="2205.14135", year=2022, authors="Dao et al.",
        title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        claim="Tiling attention to avoid materializing the N^2 matrix makes it exact, faster, and O(N) in memory.",
        tags=["attention", "efficiency", "systems"]),
    "dao2023flashattention2": dict(
        arxiv="2307.08691", year=2023, authors="Dao",
        title="FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
        claim="Better work partitioning raises attention utilization to ~70% of peak.",
        tags=["attention", "efficiency", "systems"]),

    # --- feed-forward ------------------------------------------------------
    "shazeer2020glu": dict(
        arxiv="2002.05202", year=2020, authors="Shazeer",
        title="GLU Variants Improve Transformer",
        claim="Gated activations (SwiGLU) beat ReLU/GELU FFNs at matched parameter count.",
        tags=["ffn", "architecture"]),
    "fedus2021switch": dict(
        arxiv="2101.03961", year=2021, authors="Fedus et al.",
        title="Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
        claim="Sparse expert routing decouples parameter count from per-token FLOPs.",
        tags=["ffn", "sparsity", "scaling"]),

    # --- optimization ------------------------------------------------------
    "kingma2014adam": dict(
        arxiv="1412.6980", year=2014, authors="Kingma et al.",
        title="Adam: A Method for Stochastic Optimization",
        claim="Per-parameter adaptive moments give robust convergence with little tuning.",
        tags=["optimization"]),
    "loshchilov2017adamw": dict(
        arxiv="1711.05101", year=2017, authors="Loshchilov et al.",
        title="Decoupled Weight Decay Regularization",
        claim="Decoupling weight decay from the adaptive gradient restores its regularizing effect in Adam.",
        tags=["optimization", "regularization"]),
    "loshchilov2016sgdr": dict(
        arxiv="1608.03983", year=2016, authors="Loshchilov et al.",
        title="SGDR: Stochastic Gradient Descent with Warm Restarts",
        claim="Cosine annealing of the learning rate improves final quality over step decay.",
        tags=["optimization", "schedule"]),
    "hu2024minicpm": dict(
        arxiv="2404.06395", year=2024, authors="Hu et al.",
        title="MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies",
        claim="Warmup-Stable-Decay beats cosine and makes the token budget a free choice rather than a schedule input.",
        tags=["optimization", "schedule", "small-models"]),
    "pascanu2012clipping": dict(
        arxiv="1211.5063", year=2012, authors="Pascanu et al.",
        title="On the difficulty of training Recurrent Neural Networks",
        claim="Rescaling gradients whose norm exceeds a threshold prevents exploding-gradient blowups.",
        tags=["optimization", "stability"]),
    "liu2019radam": dict(
        arxiv="1908.03265", year=2019, authors="Liu et al.",
        title="On the Variance of the Adaptive Learning Rate and Beyond",
        claim="Early adaptive-LR variance is the reason warmup is needed; longer warmup substitutes for it.",
        tags=["optimization", "schedule", "stability"]),
    "liu2025muon": dict(
        arxiv="2502.16982", year=2025, authors="Liu et al.",
        title="Muon is Scalable for LLM Training",
        claim="Orthogonalized momentum updates on 2-D weights roughly double token efficiency versus AdamW.",
        tags=["optimization", "sota"]),
    "jordan2024muon": dict(
        arxiv=None, year=2024, authors="Jordan et al.",
        url="https://kellerjordan.github.io/posts/muon/",
        title="Muon: An optimizer for hidden layers in neural networks",
        claim="A Newton-Schulz orthogonalization of the momentum buffer is a better-conditioned update direction.",
        tags=["optimization", "sota"]),
    "shazeer2018adafactor": dict(
        arxiv="1804.04235", year=2018, authors="Shazeer et al.",
        title="Adafactor: Adaptive Learning Rates with Sublinear Memory Cost",
        claim="Factored second moments cut optimizer memory from O(nm) to O(n+m).",
        tags=["optimization", "memory"]),
    "mccandlish2018batch": dict(
        arxiv="1812.06162", year=2018, authors="McCandlish et al.",
        title="An Empirical Model of Large-Batch Training",
        claim="The gradient-noise scale predicts the batch size beyond which more parallelism stops helping.",
        tags=["optimization", "batch-size", "methodology"]),
    "yang2022mup": dict(
        arxiv="2203.03466", year=2022, authors="Yang et al.",
        title="Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer",
        claim="Under muP parameterization the optimal LR is width-invariant, so it can be tuned on a small proxy.",
        tags=["optimization", "scaling", "methodology"]),
    "micikevicius2017mixed": dict(
        arxiv="1710.03740", year=2017, authors="Micikevicius et al.",
        title="Mixed Precision Training",
        claim="Half-precision compute with a full-precision master copy matches fp32 quality at ~2x throughput.",
        tags=["precision", "systems"]),

    # --- objective / regularization ---------------------------------------
    "chowdhery2022palm": dict(
        arxiv="2204.02311", year=2022, authors="Chowdhery et al.",
        title="PaLM: Scaling Language Modeling with Pathways",
        claim="An auxiliary z-loss on log-sum-exp of logits keeps the softmax normalizer near 1 and prevents divergence.",
        tags=["objective", "stability"]),
    "gemma2024gemma2": dict(
        arxiv="2408.00118", year=2024, authors="Gemma Team et al.",
        title="Gemma 2: Improving Open Language Models at a Practical Size",
        claim="Soft-capping attention and final logits with tanh bounds activations without hard clipping.",
        tags=["objective", "stability"]),
    "szegedy2015labelsmooth": dict(
        arxiv="1512.00567", year=2015, authors="Szegedy et al.",
        title="Rethinking the Inception Architecture for Computer Vision",
        claim="Label smoothing regularizes overconfident softmax outputs and improves generalization.",
        tags=["regularization", "calibration"]),
    "srivastava2014dropout": dict(
        arxiv=None, year=2014, authors="Srivastava et al.",
        url="https://jmlr.org/papers/v15/srivastava14a.html",
        title="Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
        claim="Randomly zeroing units approximates an ensemble and reduces overfitting.",
        tags=["regularization"]),
    "press2016tying": dict(
        arxiv="1608.05859", year=2016, authors="Press et al.",
        title="Using the Output Embedding to Improve Language Models",
        claim="Tying input and output embeddings cuts parameters and lowers perplexity.",
        tags=["architecture", "efficiency"]),
    "gloeckle2024mtp": dict(
        arxiv="2404.19737", year=2024, authors="Gloeckle et al.",
        title="Better & Faster Large Language Models via Multi-token Prediction",
        claim="Predicting several future tokens per position improves sample efficiency and enables self-speculation.",
        tags=["objective", "future-work"]),

    # --- scaling / methodology --------------------------------------------
    "kaplan2020scaling": dict(
        arxiv="2001.08361", year=2020, authors="Kaplan et al.",
        title="Scaling Laws for Neural Language Models",
        claim="Loss is a smooth power law in parameters, data and compute; shape details matter far less than scale.",
        tags=["scaling", "methodology"]),
    "hoffmann2022chinchilla": dict(
        arxiv="2203.15556", year=2022, authors="Hoffmann et al.",
        title="Training Compute-Optimal Large Language Models",
        claim="For a fixed compute budget, parameters and training tokens should scale together (~20 tokens/param).",
        tags=["scaling", "data", "methodology"]),
    "henighan2020scaling": dict(
        arxiv="2010.14701", year=2020, authors="Henighan et al.",
        title="Scaling Laws for Autoregressive Generative Modeling",
        claim="Power-law scaling holds across modalities; bits-per-byte is the scale-free loss unit.",
        tags=["scaling", "evaluation"]),

    # --- data / tokenization ----------------------------------------------
    "sennrich2015bpe": dict(
        arxiv="1508.07909", year=2015, authors="Sennrich et al.",
        title="Neural Machine Translation of Rare Words with Subword Units",
        claim="Byte-pair merges give an open vocabulary with a fixed-size symbol table.",
        tags=["tokenization"]),
    "eldan2023tinystories": dict(
        arxiv="2305.07759", year=2023, authors="Eldan et al.",
        title="TinyStories: How Small Can Language Models Be and Still Speak Coherent English?",
        claim="A narrow, high-quality corpus lets <10M-parameter models produce fluent, consistent English.",
        tags=["data", "small-models"]),
    "gebru2018datasheets": dict(
        arxiv="1803.09010", year=2018, authors="Gebru et al.",
        title="Datasheets for Datasets",
        claim="Datasets should ship provenance, composition, and intended-use documentation.",
        tags=["governance", "data"]),
    "mitchell2018modelcards": dict(
        arxiv="1810.03993", year=2018, authors="Mitchell et al.",
        title="Model Cards for Model Reporting",
        claim="Models should ship disaggregated evaluation and explicit scope/limitation statements.",
        tags=["governance", "evaluation"]),

    # --- alignment / chat --------------------------------------------------
    "ouyang2022instructgpt": dict(
        arxiv="2203.02155", year=2022, authors="Ouyang et al.",
        title="Training language models to follow instructions with human feedback",
        claim="Supervised instruction tuning plus preference optimization aligns outputs with user intent.",
        tags=["alignment", "chat"]),
    "wang2022selfinstruct": dict(
        arxiv="2212.10560", year=2022, authors="Wang et al.",
        title="Self-Instruct: Aligning Language Models with Self-Generated Instructions",
        claim="Instruction data can be bootstrapped from templates/models instead of collected by hand.",
        tags=["alignment", "data"]),
    "hu2021lora": dict(
        arxiv="2106.09685", year=2021, authors="Hu et al.",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        claim="Low-rank adapters match full fine-tuning at a fraction of trainable parameters.",
        tags=["finetuning", "efficiency"]),
    "touvron2023llama": dict(
        arxiv="2302.13971", year=2023, authors="Touvron et al.",
        title="LLaMA: Open and Efficient Foundation Language Models",
        claim="RMSNorm + SwiGLU + RoPE + pre-norm is the strong open recipe at every size.",
        tags=["architecture", "recipe"]),
    "touvron2023llama2": dict(
        arxiv="2307.09288", year=2023, authors="Touvron et al.",
        title="Llama 2: Open Foundation and Fine-Tuned Chat Models",
        claim="GQA plus staged SFT/RLHF yields deployable chat models.",
        tags=["architecture", "chat", "recipe"]),
    "jiang2023mistral": dict(
        arxiv="2310.06825", year=2023, authors="Jiang et al.",
        title="Mistral 7B",
        claim="GQA + sliding-window attention beats larger dense models at lower inference cost.",
        tags=["architecture", "inference", "recipe"]),

    # --- inference / deployment -------------------------------------------
    "holtzman2019nucleus": dict(
        arxiv="1904.09751", year=2019, authors="Holtzman et al.",
        title="The Curious Case of Neural Text Degeneration",
        claim="Maximization decoding produces degenerate repetition; nucleus (top-p) sampling fixes it.",
        tags=["decoding", "evaluation"]),
    "leviathan2022speculative": dict(
        arxiv="2211.17192", year=2022, authors="Leviathan et al.",
        title="Fast Inference from Transformers via Speculative Decoding",
        claim="A small draft model plus verification yields 2-3x decode speedup with identical output distribution.",
        tags=["decoding", "inference", "future-work"]),
    "kwon2023pagedattention": dict(
        arxiv="2309.06180", year=2023, authors="Kwon et al.",
        title="Efficient Memory Management for Large Language Model Serving with PagedAttention",
        claim="Paging the KV cache removes fragmentation and raises serving throughput several-fold.",
        tags=["inference", "systems", "future-work"]),
    "shoeybi2019megatron": dict(
        arxiv="1909.08053", year=2019, authors="Shoeybi et al.",
        title="Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism",
        claim="Tensor-parallel layers let a single transformer span many devices.",
        tags=["systems", "scaling", "future-work"]),

    # --- automated search / methodology ------------------------------------
    "zoph2016nas": dict(
        arxiv="1611.01578", year=2016, authors="Zoph et al.",
        title="Neural Architecture Search with Reinforcement Learning",
        claim="Architectures can be discovered by search rather than hand design.",
        tags=["autoresearch", "search"]),
    "li2016hyperband": dict(
        arxiv="1603.06560", year=2016, authors="Li et al.",
        title="Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization",
        claim="Successive halving on cheap partial runs beats full-fidelity random search per unit compute.",
        tags=["autoresearch", "search", "methodology"]),
    "lu2024aiscientist": dict(
        arxiv="2408.06292", year=2024, authors="Lu et al.",
        title="The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery",
        claim="An agent can propose, run, and write up ML experiments end to end at low cost per idea.",
        tags=["autoresearch", "methodology"]),
    "gu2023mamba": dict(
        arxiv="2312.00752", year=2023, authors="Gu et al.",
        title="Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        claim="Selective state-space layers match transformer quality with linear-time sequence scaling.",
        tags=["architecture", "future-work"]),
    "wirth2000crispdm": dict(
        arxiv=None, year=2000, authors="Wirth & Hipp",
        url="https://www.cs.unibo.it/~danilo.montesi/CBD/Beatriz/10.1.1.198.5133.pdf",
        title="CRISP-DM: Towards a Standard Process Model for Data Mining",
        claim="Data mining projects follow six iterative phases from business understanding to deployment.",
        tags=["process", "methodology"]),
}


def paper_url(key: str) -> str:
    p = PAPERS[key]
    return p.get("url") or f"https://arxiv.org/abs/{p['arxiv']}"


def cite(key: str) -> str:
    p = PAPERS[key]
    tag = f"arXiv:{p['arxiv']}" if p.get("arxiv") else p.get("url", "")
    return f"{p['authors']} ({p['year']}). {p['title']}. {tag}"


# --------------------------------------------------------------------------
# Intervention search space
# --------------------------------------------------------------------------
@dataclass
class Intervention:
    """One testable change, tied to the paper(s) that proposed it.

    `on` is applied when the hill-climber accepts the intervention; `off` is the
    ablated setting used by the baseline. Keeping both explicit means a trial is
    a genuine A/B and not just "a different config".
    """
    key: str
    title: str
    category: str
    on: dict[str, Any]
    off: dict[str, Any]
    papers: list[str]
    claim: str
    expect: str                       # what we predict to see if the claim holds
    cost: str = "neutral"             # cheaper | neutral | costlier (compute)
    risk: str = "low"
    depends_on: list[str] = field(default_factory=list)

    def paper_records(self) -> list[dict[str, Any]]:
        out = []
        for k in self.papers:
            p = dict(PAPERS[k])
            p.update(key=k, url=paper_url(k), citation=cite(k))
            out.append(p)
        return out


SEARCH_SPACE: list[Intervention] = [
    Intervention(
        key="rmsnorm", title="RMSNorm instead of LayerNorm", category="normalization",
        on={"model.norm": "rmsnorm"}, off={"model.norm": "layernorm"},
        papers=["zhang2019rmsnorm", "ba2016layernorm"],
        claim="Removing mean-centering matches LayerNorm quality at lower cost.",
        expect="val loss within noise; measurably higher tokens/s.", cost="cheaper"),
    Intervention(
        key="rope", title="Rotary position embeddings", category="positional",
        on={"model.pos": "rope"}, off={"model.pos": "learned"},
        papers=["su2021rope", "peng2023yarn"],
        claim="Relative position injected by rotating q/k beats a learned absolute table.",
        expect="lower val loss and no learned position parameters.", cost="cheaper"),
    Intervention(
        key="swiglu", title="SwiGLU feed-forward", category="ffn",
        on={"model.ffn": "swiglu"}, off={"model.ffn": "gelu"},
        papers=["shazeer2020glu", "touvron2023llama"],
        claim="Gated FFNs beat GELU FFNs at matched parameter count.",
        expect="lower val loss at ~equal parameters (d_ff shrinks to 8/3*d).", cost="neutral"),
    Intervention(
        key="gqa", title="Grouped-query attention", category="attention",
        on={"model.n_kv_head": 2}, off={"model.n_kv_head": 0},   # 0 -> full MHA
        papers=["ainslie2023gqa", "shazeer2019mqa", "jiang2023mistral"],
        claim="Sharing K/V across head groups keeps quality while shrinking the KV cache.",
        expect="val loss within noise; 3x smaller KV cache and faster decode.", cost="cheaper"),
    Intervention(
        key="qk_norm", title="Query-key normalization", category="normalization",
        on={"model.qk_norm": True}, off={"model.qk_norm": False},
        papers=["henry2020qknorm", "dehghani2023vit22b", "wortsman2023proxies"],
        claim="Normalizing q,k bounds attention logits and prevents attention entropy collapse.",
        expect="lower max attention logit; enables a higher stable learning rate.", risk="low"),
    Intervention(
        key="tie_embeddings", title="Tied input/output embeddings", category="architecture",
        on={"model.tie_embeddings": True}, off={"model.tie_embeddings": False},
        papers=["press2016tying"],
        claim="Sharing the embedding matrix with the output head cuts parameters and helps perplexity.",
        expect="fewer parameters; equal or better val loss.", cost="cheaper"),
    Intervention(
        key="resid_init", title="1/sqrt(2L) residual-projection init", category="initialization",
        on={"model.scale_resid_init": True}, off={"model.scale_resid_init": False},
        papers=["radford2019gpt2", "xiong2020prenorm"],
        claim="Down-scaling residual-branch output init keeps activation variance constant with depth.",
        expect="smaller early grad-norm spike; faster early loss drop."),
    Intervention(
        key="z_loss", title="Softmax z-loss", category="objective",
        on={"model.z_loss": 1e-4}, off={"model.z_loss": 0.0},
        papers=["chowdhery2022palm", "wortsman2023proxies"],
        claim="Penalizing log-Z^2 keeps the softmax normalizer near 1 and stops logit drift.",
        expect="bounded logit norm; equal or slightly better loss, better calibration."),
    Intervention(
        key="logit_cap", title="tanh logit soft-capping", category="objective",
        on={"model.logit_soft_cap": 30.0}, off={"model.logit_soft_cap": 0.0},
        papers=["gemma2024gemma2"],
        claim="Soft-capping final logits bounds them smoothly without hard clipping.",
        expect="lower max logit; loss within noise.", risk="medium"),
    Intervention(
        key="muon", title="Muon optimizer for hidden weights", category="optimization",
        on={"train.optimizer": "muon"}, off={"train.optimizer": "adamw"},
        papers=["liu2025muon", "jordan2024muon", "loshchilov2017adamw"],
        claim="Orthogonalized momentum updates roughly double token efficiency versus AdamW.",
        expect="clearly lower val loss at the same step budget.", cost="costlier", risk="medium"),
    Intervention(
        key="wsd", title="Warmup-Stable-Decay schedule", category="schedule",
        on={"train.schedule": "wsd"}, off={"train.schedule": "cosine"},
        papers=["hu2024minicpm", "loshchilov2016sgdr"],
        claim="A constant-LR body with a short final decay matches or beats cosine and is budget-agnostic.",
        expect="equal or lower final val loss; loss plateau then sharp end-of-run drop."),
    Intervention(
        key="long_warmup", title="Longer LR warmup", category="schedule",
        on={"train.warmup_steps": 200}, off={"train.warmup_steps": 50},
        papers=["liu2019radam", "xiong2020prenorm"],
        claim="Warmup compensates for high adaptive-LR variance in the first few hundred steps.",
        expect="lower early grad-norm; more stable start."),
    Intervention(
        key="hi_lr", title="Raise peak learning rate 2x", category="optimization",
        on={"train.lr": 6e-3}, off={"train.lr": 3e-3},
        papers=["yang2022mup", "mccandlish2018batch"],
        claim="Small models tolerate much higher LR than large-model defaults suggest; tune on the proxy.",
        expect="lower val loss if stable, divergence if not - this is a genuine risk trial.",
        risk="high"),
    Intervention(
        key="no_dropout", title="Disable dropout", category="regularization",
        on={"model.dropout": 0.0}, off={"model.dropout": 0.1},
        papers=["srivastava2014dropout", "hoffmann2022chinchilla"],
        claim="In the data-rich / few-epoch regime, dropout costs capacity without preventing overfit.",
        expect="lower val loss when the train/val gap is already small.", cost="cheaper"),
    Intervention(
        key="wd_on_2d_only", title="Weight decay only on matrices", category="regularization",
        on={"train.weight_decay": 0.1}, off={"train.weight_decay": 0.0},
        papers=["loshchilov2017adamw"],
        claim="Decoupled decay applied to 2-D weights (not norms/biases/embeddings) regularizes correctly.",
        expect="smaller train/val gap at equal train loss."),
    Intervention(
        key="rope_theta", title="Raise RoPE base to 50k", category="positional",
        on={"model.rope_theta": 50000.0}, off={"model.rope_theta": 10000.0},
        papers=["peng2023yarn", "su2021rope"],
        claim="A larger rotary base lowers frequency aliasing and eases later context extension.",
        expect="neutral at 512 ctx - included to demonstrate a true null result.",
        depends_on=["rope"]),
    Intervention(
        key="label_smoothing", title="Label smoothing 0.05", category="regularization",
        on={"train.label_smoothing": 0.05}, off={"train.label_smoothing": 0.0},
        papers=["szegedy2015labelsmooth"],
        claim="Smoothing targets curbs overconfidence and improves calibration.",
        expect="better ECE, slightly worse raw CE - a deliberate metric trade-off.", risk="medium"),
]

SEARCH_SPACE_BY_KEY = {i.key: i for i in SEARCH_SPACE}

# Baseline = deliberately a 2019-era GPT-2-style transformer, so that a
# successful hill climb *rediscovers* the modern (LLaMA-style) recipe from the
# literature rather than starting from it.
BASELINE_OVERRIDES: dict[str, Any] = {k: v for iv in SEARCH_SPACE for k, v in iv.off.items()}


def papers_for_dashboard() -> list[dict[str, Any]]:
    used: dict[str, list[str]] = {}
    for iv in SEARCH_SPACE:
        for pk in iv.papers:
            used.setdefault(pk, []).append(iv.key)
    out = []
    for k, p in PAPERS.items():
        # `p` may already carry a `url` (non-arXiv works); paper_url resolves
        # either case, so build the record then overwrite rather than **-merging.
        rec = dict(p)
        rec.update(key=k, url=paper_url(k), citation=cite(k), used_by=used.get(k, []))
        out.append(rec)
    return sorted(out, key=lambda r: (-r["year"], r["authors"]))
