# Lab 4 - Neural Models: Learning, Depth, Activations, and Output Layers

This lab studies a tiny XOR problem to connect neural-network representation, backpropagation, activations, output/loss pairing, symmetry, and validation.

## Contents

- `lab_sheet.pdf` - original laboratory sheet
- `xor_lab.py` - complete PyTorch implementation and experiments
- `test_lab.py` - automated checks for the main claims
- `llm_prompt.txt` - behavioural specification used to generate the implementation
- `report.pdf` - completed short report with results and reflection answers
- `results/` - machine-readable and human-readable experiment outputs

## Main experiments

1. Linear affine + sigmoid baseline on XOR
2. 2-2-1 XOR network with sigmoid, tanh, and ReLU hidden activations
3. Zero-initialisation symmetry experiment
4. Three-class output extension with softmax probabilities / CrossEntropyLoss
5. Softmax shift-invariance diagnostic

## Run

```bash
python3 xor_lab.py
```

## Test

```bash
python3 -m unittest -v
```

CPU is sufficient for this lab.
