# Bayesian Networks and Autoregressive Language Models Lab

This repository implements the laboratory assignment using only the Python standard library.

## Contents

- `language_models.py` - first-order and second-order autoregressive models.
- `run_lab.py` - conditional tables, normalization checks, complete parameter/context counts, and reproducible generation.
- `answers.md` - answers to Questions 1-14, numerical model comparison, and explicit context-universe definition.
- `generated_examples.txt` - 20 sampled sentences per model and five greedy runs per model.
- `lab_output.txt` - generated probability tables and numerical checks.
- `reflection.md` - implementation review and validation reflection.

## Run

```bash
python3 -m unittest -v test_language_models.py
python3 run_lab.py
```

The second-order parameter count includes both trigram-successor probabilities and the separately stored initial-token distribution. The zero-context count is reported only after defining the valid two-token context universe in `answers.md`.

The implementation uses transition counts for `P(X_t | X_{t-1})`, trigram/context counts for `P(X_t | X_{t-2}, X_{t-1})`, deterministic greedy decoding, and probabilistic sampling. No machine-learning library or pretrained language model is used.
