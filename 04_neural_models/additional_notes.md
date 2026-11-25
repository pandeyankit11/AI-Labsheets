# Viva / Submission Notes

## Key one-line answers

- Why can't one affine layer solve XOR? XOR is not linearly separable.
- Why is nonlinear hidden activation needed? Affine layers compose to one affine map.
- Why use BCEWithLogitsLoss? Binary target with a stable sigmoid + BCE formulation.
- What does `.backward()` do? Computes reverse-mode gradients through the autograd graph.
- What does `parameter.grad` mean? Gradient of the scalar loss with respect to that parameter.
- Why does zero initialisation fail? Symmetry: hidden units compute the same thing and receive the same gradient; with output weights zero, first-layer gradients are zero.
- Why does softmax sum to 1? It normalises exponentials by their total.
- Why is `p-y` the multiclass logit gradient? It is the simplified derivative of softmax cross-entropy.
- Why subtract max logit? Numerical stability; it leaves softmax probabilities unchanged.
