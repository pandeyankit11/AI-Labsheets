# Bayesian Networks and Autoregressive Language Models Lab

This repository implements the laboratory assignment using only the Python standard library.

## Contents

- `language_models.py` - first-order and second-order autoregressive models.
- `run_lab.py` - conditional tables, normalization checks, complete parameter/context counts, and reproducible generation.
- `answers.md` - answers to Questions 1-14, numerical model comparison, generation diversity, and explicit context-universe definition.
- `generated_examples.txt` - 20 sampled outputs per model and five greedy runs per model.
- `lab_output.txt` - generated probability tables and numerical checks.
- `test_language_models.py` and `test_results.txt` - independent tests and captured output.
- `reflection.md` - implementation review and inspected code correction.

## Run

```bash
python3 -m unittest -v test_language_models.py
python3 run_lab.py
```

The second-order parameter count includes both trigram-successor probabilities and the separately stored initial-token distribution. The zero-context count is reported only after defining the valid two-token context universe in `answers.md`.

The first-order greedy example is capped because strict argmax decoding loops on this corpus; this limitation is documented rather than hidden by forcing an end token or changing the decoding rule.
