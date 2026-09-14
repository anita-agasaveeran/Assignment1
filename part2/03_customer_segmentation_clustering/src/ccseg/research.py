"""Research provenance: every method used, the paper it comes from, and the line of
code that implements it.

This exists so the dashboard can be audited rather than believed. Each entry names
a concrete implementation site, so a reviewer can go from a claim on a panel to the
function that produces the number. Where this project deliberately departs from a
paper, the departure is recorded in ``deviation`` - that is the entry that matters
most, and the one usually missing.
"""
from __future__ import annotations

CITATIONS: list[dict] = [
    # ---------------------------------------------------------------- process --
    {
        "key": "crisp-dm",
        "authors": "Wirth, R. & Hipp, J.",
        "year": 1999,
        "title": "CRISP-DM: Towards a Standard Process Model for Data Mining",
        "venue": "Proc. 4th Int. Conf. Practical Applications of Knowledge Discovery and Data Mining",
        "topic": "process",
        "used_for": "The six-phase structure of the whole project and of the dashboard.",
        "implemented_in": "package layout; config.BUSINESS_CHARTER, data.py, prepare.py, "
                          "autoresearch.py, evaluate.py, deploy.py",
        "deviation": "The phase loop is executed once end-to-end and logged, rather than "
                     "iterated informally. Every backward arrow that would normally be tacit "
                     "(a data-quality finding changing the preparation step) is recorded as an "
                     "explicit resolution in the audit ledger.",
    },
    # ------------------------------------------------------------ CASH / AutoML --
    {
        "key": "auto-weka",
        "authors": "Thornton, C., Hutter, F., Hoos, H. H. & Leyton-Brown, K.",
        "year": 2013,
        "title": "Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms",
        "venue": "KDD 2013",
        "topic": "automl",
        "used_for": "The CASH formulation - treating algorithm choice as one more dimension of "
                    "the hyperparameter space rather than an outer loop.",
        "implemented_in": "search_space.SPACE (algorithm is a searched dimension); "
                          "search_space.CONDITIONS encodes the conditional structure",
        "deviation": "Supervised CASH has a loss with an unbiased held-out estimate. Clustering "
                     "has neither, so the objective is an internal index and the whole result "
                     "rests on the index being a reasonable proxy - a substantially weaker "
                     "guarantee, and the reason the evaluation phase leans on stability instead.",
    },
    {
        "key": "automl4clust",
        "authors": "Tschechlov, D., Fritz, M. & Schwarz, H.",
        "year": 2021,
        "title": "AutoML4Clust: Efficient AutoML for Clustering Analyses",
        "venue": "EDBT 2021",
        "topic": "automl",
        "used_for": "Carrying CASH to unsupervised learning; optimising CVIs (SIL, CH, DB) as the "
                    "objective; and the subsample-then-refit cost control.",
        "implemented_in": "autoresearch.Evaluator (search runs on SearchConfig.search_subsample, "
                          "champion refit on the full book); metrics.score_partition",
        "deviation": "The paper compares random search, Bayesian optimisation, Hyperband and "
                     "BOHB, and searches algorithm + k. This project searches the preprocessing "
                     "chain as well - on heavy-tailed financial data the transform choice moves "
                     "the objective more than the algorithm does - and uses hill climbing with "
                     "random search as the control arm.",
    },
    {
        "key": "ml2dac",
        "authors": "Treder-Tschechlov, D., Fritz, M., Schwarz, H. & Mitschang, B.",
        "year": 2023,
        "title": "ML2DAC: Meta-Learning to Democratize AutoML for Clustering Analysis",
        "venue": "SIGMOD 2023",
        "topic": "automl",
        "used_for": "The argument that CVI *selection* is itself a decision that biases the "
                    "result, which motivates the calibrated composite over a single index.",
        "implemented_in": "metrics.ObjectiveCalibration, metrics.COMPOSITE_WEIGHTS",
        "deviation": "ML2DAC learns which CVI to use from a meta-knowledge base of prior "
                     "datasets. No such base exists here, so the three indices are combined "
                     "with fixed, stated weights instead of selected - simpler, and honest "
                     "about the fact that the weighting is a judgement.",
    },
    {
        "key": "fanova",
        "authors": "Hutter, F., Hoos, H. & Leyton-Brown, K.",
        "year": 2014,
        "title": "An Efficient Approach for Assessing Hyperparameter Importance",
        "venue": "ICML 2014",
        "topic": "automl",
        "used_for": "Attributing objective variance to individual search dimensions - which "
                    "decisions in the pipeline actually mattered.",
        "implemented_in": "autoresearch.dimension_importance",
        "deviation": "fANOVA fits a random forest over the space and marginalises properly. "
                     "This is a one-way eta-squared over observed trials, which is cheaper and "
                     "biased by the optimiser's own sampling. The random-search arm is pooled in "
                     "to dilute that bias, and the panel states the caveat rather than hiding it.",
    },
    # ------------------------------------------------------- search heuristics --
    {
        "key": "tabu",
        "authors": "Glover, F.",
        "year": 1986,
        "title": "Future Paths for Integer Programming and Links to Artificial Intelligence",
        "venue": "Computers & Operations Research 13(5)",
        "topic": "search",
        "used_for": "Short-term memory preventing the climber from cycling on a plateau.",
        "implemented_in": "autoresearch.hill_climb (deque `tabu`, SearchConfig.tabu_size)",
        "deviation": "Attribute-level tabu tenure is replaced by whole-configuration hashing, "
                     "which is coarser but exact for a conditional space where 'the same "
                     "attribute' is not always well defined.",
    },
    {
        "key": "gsat",
        "authors": "Selman, B., Levesque, H. & Mitchell, D.",
        "year": 1992,
        "title": "A New Method for Solving Hard Satisfiability Problems",
        "venue": "AAAI 1992",
        "topic": "search",
        "used_for": "Sideways (non-worsening) moves as the mechanism for crossing plateaus, and "
                    "random restarts as the mechanism for escaping local optima.",
        "implemented_in": "autoresearch.hill_climb (accept_sideways branch, SearchConfig.restarts)",
        "deviation": "Plateaus here come from categorical dimensions with no effect on the "
                     "current pipeline (e.g. changing `init` under a non-KMeans algorithm), so "
                     "the sideways budget is small and per-restart.",
    },
    {
        "key": "vns",
        "authors": "Mladenovic, N. & Hansen, P.",
        "year": 1997,
        "title": "Variable Neighborhood Search",
        "venue": "Computers & Operations Research 24(11)",
        "topic": "search",
        "used_for": "Biasing neighbourhood sampling toward dimensions that have historically "
                    "produced improvements.",
        "implemented_in": "autoresearch.hill_climb (`dim_credit`), search_space.neighbours(dim_weights=)",
        "deviation": "Formal VNS systematically enlarges the neighbourhood radius; here the "
                     "radius stays at one dimension and only the sampling distribution adapts.",
    },
    {
        "key": "constraints",
        "authors": "Coello Coello, C. A.",
        "year": 2002,
        "title": "Theoretical and Numerical Constraint-Handling Techniques used with Evolutionary Algorithms",
        "venue": "Computer Methods in Applied Mechanics and Engineering 191(11-12)",
        "topic": "search",
        "used_for": "Static-penalty constraint handling, so that any feasible solution strictly "
                    "dominates any infeasible one while infeasible ones keep a gradient.",
        "implemented_in": "metrics.INFEASIBLE_SHIFT, metrics.score_partition",
        "deviation": "None material. The shift is set to the maximum attainable raw objective "
                     "(1.0), which makes the ordering exactly lexicographic.",
    },
    # ------------------------------------------------------------------ indices --
    {
        "key": "silhouette",
        "authors": "Rousseeuw, P. J.",
        "year": 1987,
        "title": "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis",
        "venue": "Journal of Computational and Applied Mathematics 20",
        "topic": "validity",
        "used_for": "Primary internal index and the per-sample silhouette diagnostic.",
        "implemented_in": "metrics.internal_indices, metrics.silhouette_profile",
        "deviation": "Computed on a seeded subsample above 2,500 points (the index is O(n^2)). "
                     "The dashboard shows the per-cluster distribution, not only the mean, "
                     "because that is what Rousseeuw's original diagnostic is for.",
    },
    {
        "key": "ch",
        "authors": "Calinski, T. & Harabasz, J.",
        "year": 1974,
        "title": "A Dendrite Method for Cluster Analysis",
        "venue": "Communications in Statistics 3(1)",
        "topic": "validity",
        "used_for": "Variance-ratio criterion; second component of the composite objective.",
        "implemented_in": "metrics.internal_indices, metrics.ObjectiveCalibration",
        "deviation": "CH is unbounded, so it cannot be averaged with a bounded index directly. "
                     "It is mapped through ch/(ch+kappa) with kappa frozen from a calibration "
                     "draw before the search, keeping the objective stationary.",
    },
    {
        "key": "db",
        "authors": "Davies, D. L. & Bouldin, D. W.",
        "year": 1979,
        "title": "A Cluster Separation Measure",
        "venue": "IEEE TPAMI 1(2)",
        "topic": "validity",
        "used_for": "Within-to-between scatter ratio; third component of the composite.",
        "implemented_in": "metrics.internal_indices",
        "deviation": "Inverted to 1/(1+DB) so that, like the others, larger is better.",
    },
    {
        "key": "arbelaitz",
        "authors": "Arbelaitz, O., Gurrutxaga, I., Muguerza, J., Perez, J. M. & Perona, I.",
        "year": 2013,
        "title": "An Extensive Comparative Study of Cluster Validity Indices",
        "venue": "Pattern Recognition 46(1)",
        "topic": "validity",
        "used_for": "Justifying *which* three indices to use, rather than picking the familiar one.",
        "implemented_in": "metrics module docstring; choice of SIL / CH / DB",
        "deviation": "The study evaluates 30 indices on data with known ground truth. No ground "
                     "truth exists here, so its ranking is imported as prior evidence, not "
                     "re-derived.",
    },
    {
        "key": "milligan",
        "authors": "Milligan, G. W. & Cooper, M. C.",
        "year": 1985,
        "title": "An Examination of Procedures for Determining the Number of Clusters in a Data Set",
        "venue": "Psychometrika 50(2)",
        "topic": "validity",
        "used_for": "The documented bias of stopping rules toward small k - the reason the "
                    "objective is constrained and the k-sweep is reported in full.",
        "implemented_in": "evaluate.k_sweep; config.SearchConfig.min_k_business",
        "deviation": "Their comparison is on well-separated synthetic data. On this book the "
                     "bias is confirmed empirically in the k-sweep panel rather than assumed.",
    },
    {
        "key": "vendramin",
        "authors": "Vendramin, L., Campello, R. J. G. B. & Hruschka, E. R.",
        "year": 2010,
        "title": "Relative Clustering Validity Criteria: A Comparative Overview",
        "venue": "Statistical Analysis and Data Mining 3(4)",
        "topic": "validity",
        "used_for": "The general result that no single relative criterion dominates, motivating "
                    "a weighted combination plus constraints.",
        "implemented_in": "metrics.COMPOSITE_WEIGHTS",
        "deviation": "Weights are asserted (0.50/0.30/0.20) and stated, not learned. Sensitivity "
                     "to them is bounded by the leaderboard, which shows the top configurations "
                     "under the composite alongside their individual indices.",
    },
    # ---------------------------------------------------------------- stability --
    {
        "key": "benhur",
        "authors": "Ben-Hur, A., Elisseeff, A. & Guyon, I.",
        "year": 2002,
        "title": "A Stability Based Method for Discovering Structure in Clustered Data",
        "venue": "Pacific Symposium on Biocomputing",
        "topic": "stability",
        "used_for": "Resampling stability as the primary evidence that structure is real.",
        "implemented_in": "evaluate.stability",
        "deviation": "The paper compares two independent subsamples on their overlap. Here every "
                     "replicate is compared to the full-data reference partition, which is the "
                     "quantity actually of interest in deployment ('would this month's refit "
                     "reproduce the segments we are campaigning against?').",
    },
    {
        "key": "lange",
        "authors": "Lange, T., Roth, V., Braun, M. L. & Buhmann, J. M.",
        "year": 2004,
        "title": "Stability-Based Validation of Clustering Solutions",
        "venue": "Neural Computation 16(6)",
        "topic": "stability",
        "used_for": "Using an inductive extension of a transductive clustering to transfer labels "
                    "between resamples - exactly what deployment also requires.",
        "implemented_in": "models.InductiveAssigner (and its reported fidelity_)",
        "deviation": "They train a classifier as the extension operator; here it is a "
                     "nearest-centroid rule, which is weaker but has no hyperparameters of its "
                     "own to confound the stability estimate, and is the rule that actually ships.",
    },
    {
        "key": "vonluxburg-stability",
        "authors": "von Luxburg, U.",
        "year": 2010,
        "title": "Clustering Stability: An Overview",
        "venue": "Foundations and Trends in Machine Learning 2(3)",
        "topic": "stability",
        "used_for": "The caution that stability is not a monotone signal for k and can be high "
                    "for the wrong reasons.",
        "implemented_in": "evaluate.k_sweep (stability reported per k, never used alone)",
        "deviation": "Treated as one of three converging lines of evidence for k, alongside the "
                     "constrained objective and the gap statistic.",
    },
    {
        "key": "hennig",
        "authors": "Hennig, C.",
        "year": 2007,
        "title": "Cluster-wise Assessment of Cluster Stability",
        "venue": "Computational Statistics & Data Analysis 52(1)",
        "topic": "stability",
        "used_for": "Per-cluster bootstrap Jaccard and the 0.60 / 0.85 interpretation bands "
                    "(SC-2) - a partition can be stable on average and still contain a segment "
                    "that dissolves.",
        "implemented_in": "evaluate.stability (per_cluster_jaccard)",
        "deviation": "Subsampling without replacement rather than the bootstrap, to avoid "
                     "duplicate points inflating the apparent compactness of a cluster.",
    },
    {
        "key": "hubert",
        "authors": "Hubert, L. & Arabie, P.",
        "year": 1985,
        "title": "Comparing Partitions",
        "venue": "Journal of Classification 2(1)",
        "topic": "stability",
        "used_for": "Adjusted Rand Index for partition agreement across resamples.",
        "implemented_in": "evaluate.stability, evaluate.k_sweep",
        "deviation": "None.",
    },
    {
        "key": "vinh",
        "authors": "Vinh, N. X., Epps, J. & Bailey, J.",
        "year": 2010,
        "title": "Information Theoretic Measures for Clusterings Comparison",
        "venue": "JMLR 11",
        "topic": "stability",
        "used_for": "NMI as a second agreement measure with different bias to ARI.",
        "implemented_in": "evaluate.stability",
        "deviation": "Reported alongside ARI, not instead of it; the paper's point is precisely "
                     "that the choice of measure is not neutral.",
    },
    {
        "key": "tibshirani",
        "authors": "Tibshirani, R., Walther, G. & Hastie, T.",
        "year": 2001,
        "title": "Estimating the Number of Clusters in a Data Set via the Gap Statistic",
        "venue": "JRSS-B 63(2)",
        "topic": "validity",
        "used_for": "An independent, null-referenced read on k.",
        "implemented_in": "evaluate.gap_statistic",
        "deviation": "Uses the paper's PCA-aligned uniform reference, with a smaller number of "
                     "Monte-Carlo draws than recommended for budget reasons; s_k is reported so "
                     "the resulting uncertainty is visible.",
    },
    {
        "key": "hopkins",
        "authors": "Lawson, R. G. & Jurs, P. C.",
        "year": 1990,
        "title": "New Index for Clustering Tendency and its Application to Chemical Problems",
        "venue": "Journal of Chemical Information and Computer Sciences 30(1)",
        "topic": "validity",
        "used_for": "Testing before modelling whether the data is clusterable at all.",
        "implemented_in": "data.clustering_tendency",
        "deviation": "Sampling-based estimate over a 5% sample; the statistic is sensitive to "
                     "the sampling window, so it is read as a coarse go/no-go, not a p-value.",
    },
    # --------------------------------------------------------------- algorithms --
    {
        "key": "gmm",
        "authors": "Fraley, C. & Raftery, A. E.",
        "year": 1998,
        "title": "How Many Clusters? Which Clustering Method? Answers Via Model-Based Cluster Analysis",
        "venue": "The Computer Journal 41(8)",
        "topic": "algorithms",
        "used_for": "Model-based clustering as a family with a different inductive bias to "
                    "k-means (elliptical, unequal-covariance clusters).",
        "implemented_in": "models.build_clusterer (GaussianMixture, covariance searched)",
        "deviation": "Model selection is by the shared composite objective rather than BIC, so "
                     "that every algorithm in the zoo is judged on identical terms.",
    },
    {
        "key": "hdbscan",
        "authors": "Campello, R. J. G. B., Moulavi, D. & Sander, J.",
        "year": 2013,
        "title": "Density-Based Clustering Based on Hierarchical Density Estimates",
        "venue": "PAKDD 2013",
        "topic": "algorithms",
        "used_for": "A density family that can decline to assign points, as a check on whether "
                    "the partition-forcing families are inventing structure.",
        "implemented_in": "models.build_clusterer (sklearn.cluster.HDBSCAN)",
        "deviation": "Noise fraction is penalised in the objective: an unassigned account still "
                     "has to receive a treatment, so a solution that abstains on a third of the "
                     "book is not usable however clean the remainder looks.",
    },
    {
        "key": "spectral",
        "authors": "von Luxburg, U.",
        "year": 2007,
        "title": "A Tutorial on Spectral Clustering",
        "venue": "Statistics and Computing 17(4)",
        "topic": "algorithms",
        "used_for": "A graph-based family able to find non-convex structure.",
        "implemented_in": "models.build_clusterer (SpectralClustering); models.MAX_N guard",
        "deviation": "Guarded above n=3,000 because the eigendecomposition is super-linear; it "
                     "therefore competes only during the subsampled search, which is stated "
                     "rather than left implicit.",
    },
    {
        "key": "birch",
        "authors": "Zhang, T., Ramakrishnan, R. & Livny, M.",
        "year": 1996,
        "title": "BIRCH: An Efficient Data Clustering Method for Very Large Databases",
        "venue": "SIGMOD 1996",
        "topic": "algorithms",
        "used_for": "A single-pass hierarchical method as the scalability reference point.",
        "implemented_in": "models.build_clusterer (Birch)",
        "deviation": "None.",
    },
    # ----------------------------------------------------- interpretation / ops --
    {
        "key": "trepan",
        "authors": "Craven, M. W. & Shavlik, J. W.",
        "year": 1995,
        "title": "Extracting Tree-Structured Representations of Trained Networks",
        "venue": "NIPS 1995",
        "topic": "interpretability",
        "used_for": "The surrogate-tree pattern and, critically, the distinction between "
                    "fidelity to the model and accuracy against truth.",
        "implemented_in": "profiling.surrogate_rules (reports fidelity, SC-5)",
        "deviation": "A plain CART surrogate rather than TREPAN's oracle-queried induction; the "
                     "training set is large enough that querying an oracle for synthetic points "
                     "adds little.",
    },
    {
        "key": "tsne",
        "authors": "van der Maaten, L. & Hinton, G.",
        "year": 2008,
        "title": "Visualizing Data using t-SNE",
        "venue": "JMLR 9",
        "topic": "interpretability",
        "used_for": "Local-neighbourhood view of the segments in the scatter panel.",
        "implemented_in": "profiling.projection",
        "deviation": "Shown *beside* PCA and never alone: in t-SNE, between-cluster distance and "
                     "relative cluster size carry no meaning, and reading it alone is how "
                     "people convince themselves of clusters that are not there.",
    },
    {
        "key": "modelcards",
        "authors": "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., "
                   "Spitzer, E., Raji, I. D. & Gebru, T.",
        "year": 2019,
        "title": "Model Cards for Model Reporting",
        "venue": "FAT* 2019",
        "topic": "operations",
        "used_for": "The structure of the model card, including intended use, out-of-scope uses, "
                    "factors and ethical considerations.",
        "implemented_in": "deploy.model_card",
        "deviation": "The 'factors' section is adapted: the file contains no protected "
                     "attributes, so the section records the *proxy* risk and states that a "
                     "disparate-impact review cannot be performed from this data at all.",
    },
    {
        "key": "sculley",
        "authors": "Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., "
                   "Chaudhary, V., Young, M., Crespo, J.-F. & Dennison, D.",
        "year": 2015,
        "title": "Hidden Technical Debt in Machine Learning Systems",
        "venue": "NIPS 2015",
        "topic": "operations",
        "used_for": "Why feature engineering lives inside the scoring artifact (training/serving "
                    "skew) and why the monitor watches inputs rather than waiting for outcomes.",
        "implemented_in": "deploy.ScoringArtifact.score (calls fixed_clean internally); "
                          "deploy.drift_report",
        "deviation": "None.",
    },
]

TOPICS = ["process", "automl", "search", "validity", "stability", "algorithms",
          "interpretability", "operations"]


def by_topic() -> dict[str, list[dict]]:
    return {t: [c for c in CITATIONS if c["topic"] == t] for t in TOPICS}
