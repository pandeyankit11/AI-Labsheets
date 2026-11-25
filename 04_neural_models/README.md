# Lab 4 - Neural Models: Learning, Depth, Activations, and Output Layers

Question sheet: lab_sheet.pdf (7 pages).

## Contents
- xor_lab.py: linear XOR baseline, 2-2-1 binary XOR network, zero-initialization symmetry check, activation comparison, three-class extension, and softmax shift-invariance diagnostic.
- test_lab.py: regression checks for XOR, symmetry, three-class outputs, and saved first-layer gradient tensors.
- REPORT.md: authoritative updated write-up with the problem sketch, task explanations, gradient matrices, numerical results, and reflection answers.
- llm_prompt.txt: behavioural specification used for the implementation.
- results/: activation metrics, first-layer gradient matrices, symmetry/three-class results, and human-readable run output.
- test_results.txt: test coverage and numerical gradient evidence.
- report.pdf: original PDF snapshot retained from the earlier submission; it has not been regenerated to include the newly saved gradient-tensor appendix. Use REPORT.md as the current report.

## Run

    python3 -m pip install -r requirements.txt
    python3 xor_lab.py
    python3 -m unittest -v test_lab.py

The captured gradient tensor is the 2x2 fc1.weight.grad immediately after the first backward pass and before the optimizer update. Its Frobenius norm matches the saved scalar norm for each activation.
