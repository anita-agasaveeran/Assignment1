"""CRISP-LM: a small language model + chatbot built end-to-end under CRISP-DM.

Package layout mirrors the six CRISP-DM phases:
  data.py      -> Data Understanding + Data Preparation
  tokenizer.py -> Data Preparation
  model.py     -> Modeling (architecture)
  optim.py     -> Modeling (optimization)
  train.py     -> Modeling (fitting)
  evaluate.py  -> Evaluation
  autoresearch.py -> Modeling/Evaluation (automated hill-climbing over papers)
  serve/app    -> Deployment
"""
__version__ = "0.1.0"
