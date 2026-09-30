# Bayesian Networks and Autoregressive Language Models Lab

This repository implements the laboratory assignment using only the Python standard library.

## Contents

- `language_models.py` - first-order and second-order autoregressive models.
- `run_lab.py` - trains both models, prints conditional probability tables, checks normalization, reports greedy predictions, and generates sample text.
- `answers.md` - answers to Questions 1-14 and the requested conceptual comparisons.
- `generated_examples.txt` - reproducible generated sentences using seed 42.
- `reflection.md` - reflection on using an LLM as an implementation assistant and validating the result.

## Run

```bash
python3 run_lab.py
```

The implementation uses:

- transition counts for the first-order model: `P(X_t | X_{t-1})`
- trigram/context counts for the second-order model: `P(X_t | X_{t-2}, X_{t-1})`
- deterministic greedy generation via `argmax`
- probabilistic generation via random sampling
- explicit normalization checks

No machine-learning library or pretrained language model is used.
