# Neural Models Lab - Completed Report

## 1. Task 1 - Problem specification

### Input and output spaces

The input is a pair of binary sensor values:

- `x1 ∈ {0,1}`
- `x2 ∈ {0,1}`

The output is a binary warning `y ∈ {0,1}` indicating sensor disagreement.

| x1 | x2 | y |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

### Why one straight boundary is insufficient

The positive examples are `(0,1)` and `(1,0)`, while the negative examples are `(0,0)` and `(1,1)`. There is no single straight line that separates the two classes. Therefore a single affine decision function followed by a sigmoid cannot represent XOR exactly.

### Linear-model prediction

A single affine layer followed by a sigmoid was trained as a baseline. Its loss converged to approximately `0.693147`, with probabilities near `0.5`, and it classified only 2/4 examples correctly. This is consistent with the non-linearly separable structure of XOR.

---

## 2. Task 2 - Model design

### Binary model

Baseline architecture:

`2 inputs -> 2 hidden units -> 1 output`

Hidden activation: `tanh` for the reference design, with sigmoid and ReLU evaluated separately.

Output: one logit.

Loss: `BCEWithLogitsLoss`.

Optimizer: Adam.

Training: full-batch, CPU, 1000 steps, learning rate `0.05`, seed `2`.

### Why the hidden nonlinearity is necessary

Without a nonlinear hidden activation, multiple affine layers collapse into one affine map, so extra affine depth does not make XOR representable. The nonlinearity lets the hidden layer construct a new representation in which the final linear output can separate the classes.

### Why output logits + BCEWithLogitsLoss are paired

The target is a single binary decision. `BCEWithLogitsLoss` combines the sigmoid transformation with binary cross-entropy in a numerically stable implementation, while still representing a Bernoulli/binary prediction.

### Evidence of learning

The experiment treats all of the following as evidence:

1. initial loss decreases substantially;
2. all four thresholded predictions are correct;
3. predicted probabilities move close to the correct class values;
4. the first-layer gradient is nonzero early in training;
5. the behaviour is reproducible with a fixed seed.

---

## 3. Task 3 - LLM-assisted implementation

The exact behavioural prompt used is stored in `llm_prompt.txt`.

Before execution, the implementation was checked for:

- correct XOR dataset and labels;
- exact 2-2-1 architecture;
- nonlinear hidden layer;
- binary output representation;
- `BCEWithLogitsLoss`;
- explicit `loss.backward()` call;
- optimiser update after backpropagation;
- reproducibility through a fixed seed;
- explicit output of loss, probabilities, labels, and gradients.

Engineering adjustments enforced before execution were fixed-seed reproducibility and explicit validation/diagnostic outputs rather than trusting a single final loss value.

---

## 4. Task 4 - Execute, test, and diagnose

### 4A. Basic learning check

| Hidden activation | Initial loss | Final loss | 4/4 correct? | Early `||grad_W1||_2` |
|---|---:|---:|---|---:|
| Sigmoid | 0.698454 | 0.002016 | Yes | 0.005453 |
| Tanh | 0.693229 | 0.000688 | Yes | 0.005077 |
| ReLU | 0.693667 | 0.000280 | Yes | 0.004868 |

All three random-initialised networks learned the four XOR examples under this seed and optimisation setting.

### Final binary probabilities

Sigmoid hidden:

`[0.001776, 0.997904, 0.997879, 0.002063]`

Tanh hidden:

`[0.000538, 0.999245, 0.999059, 0.000519]`

ReLU hidden:

`[0.000157, 0.999569, 0.999569, 0.000103]`

Thresholding at 0.5 gives `[0,1,1,0]` in all three runs.

### 4B. Backpropagation check

After `loss.backward()`, `model.fc1.weight.grad` represents the gradient of the scalar batch loss with respect to the first-layer weight matrix:

`dL / dW^(1)`.

With a mean loss over the four training examples, the gradient is the mean of the corresponding example-wise gradients because differentiation is linear over the averaged loss:

`∂[(1/4)Σ_i L_i]/∂W = (1/4)Σ_i ∂L_i/∂W`.

### 4C. Symmetry experiment

All weights and biases in a 2-2-1 tanh network were initialised to zero.

Observed for every recorded step:

- hidden-layer row difference = `0`;
- first-layer gradient norm = `0`;
- loss remained `0.693147`;
- both hidden rows remained identical.

Explanation: both hidden neurons start with identical parameters, therefore they compute the same activations. With the output-layer weights also initialised to zero, the first-layer gradient is zero, so the optimiser has no signal that can break the symmetry. The two hidden units cannot spontaneously specialise into distinct features.

### 4D. Activation experiment interpretation

The three activations all solved XOR for this particular four-point experiment. Their final losses and gradient norms differ because their derivatives have different shapes and therefore produce different backpropagated learning signals.

The result should not be interpreted as a universal ranking of sigmoid, tanh, and ReLU. The assignment specifically asks for an explanation of the observed behaviour for this experiment rather than claiming a universally best activation.

---

## 5. Task 5 - Three-class extension

Classes:

- `0`: `(0,0)`
- `1`: `(0,1)` or `(1,0)`
- `2`: `(1,1)`

Architecture:

`2 inputs -> 2 tanh hidden units -> 3 logits`

Loss: `CrossEntropyLoss`.

The final output weight matrix has shape `(3,2)`, so each example produces exactly 3 logits.

### Predictions

| Input | P(class 0) | P(class 1) | P(class 2) | Prediction |
|---|---:|---:|---:|---:|
| (0,0) | 0.999786 | 0.000214 | ~0 | 0 |
| (0,1) | 0.000073 | 0.999875 | 0.000053 | 1 |
| (1,0) | 0.000065 | 0.999875 | 0.000060 | 1 |
| (1,1) | ~0 | 0.000194 | 0.999805 | 2 |

Every probability vector sums to approximately `1.0`, and all four examples are classified correctly.

### Why softmax probabilities sum to one

For logits `z_i`, softmax is:

`p_i = exp(z_i) / Σ_j exp(z_j)`.

Therefore:

`Σ_i p_i = 1`.

### Why the logit gradient is `p - y`

For softmax followed by cross-entropy, differentiation simplifies to:

`∂L/∂z = p - y`.

### Optional numerical stability diagnostic

Adding `100` to every logit changed the softmax probabilities by at most approximately `1.0e-9` in this experiment, which is effectively unchanged at normal numerical precision. This is why subtracting the maximum logit before exponentiation preserves the probability distribution while avoiding unnecessarily large exponentials.

---

## 6. Reflection answers

### 1. What did XOR demonstrate about depth and nonlinearity?

Depth alone is not enough if every layer is affine. A composition of affine maps is still affine, so it cannot represent XOR. A nonlinear hidden activation changes the representational class of the network and makes the XOR mapping representable.

### 2. What evidence showed that backpropagation supplied a useful learning signal?

A useful learning signal is more than simply a nonzero gradient. The loss decreased substantially, the four predicted labels became correct, probabilities moved close to the target classes, and the model reached the desired behaviour repeatedly under the fixed experimental configuration. The early first-layer gradient norms were also nonzero in the successful random-initialised runs.

### 3. Why did identical/zero initialisation prevent distinct hidden features?

Identical hidden units have identical forward activations and therefore receive identical gradients. With all weights and biases zero, the output-layer weights also provide no signal to move the hidden layer, so symmetry is never broken.

### 4. How did changing the hidden activation affect the gradient?

The measured early first-layer gradient norms differed slightly across sigmoid, tanh and ReLU. Scientifically, the cause is the different activation derivatives and resulting Jacobian factors. Engineering-wise, the observed result is specific to this dataset, initialisation, seed, optimizer and hyperparameters and should not be promoted into a universal ranking.

### 5. Why must output layer and loss be selected together?

The output representation and loss encode assumptions about the prediction task. A single binary target calls for a Bernoulli-style output and binary cross-entropy; a one-of-K classification task calls for multiple logits and multiclass cross-entropy. The pairing determines the probabilistic interpretation and gradient structure used for training.

### 6. Where did the LLM help and where was verification essential?

The LLM was useful for producing the initial PyTorch scaffolding and for translating the model specification into code. Human verification was essential for checking that the architecture, loss, gradient inspection, symmetry experiment, output-layer modification and probability-normalisation test actually matched the laboratory specification.

### 7. Which tests scale and which become expensive?

Loss tracking, prediction checks, probability-sum checks, and targeted gradient inspections scale well. Exhaustive finite-difference checks become expensive because each parameter requires additional forward evaluations, so they are useful mainly for debugging small models rather than large networks.

---

## 7. Main conclusion

The laboratory demonstrates the progression:

`representation -> nonlinear model -> backpropagation -> optimisation -> validation -> output/loss matching`.

The central scientific result is that XOR requires an appropriate nonlinear representation. The central engineering result is that successful code must be validated using losses, predictions, gradients, symmetry diagnostics, and task-appropriate output/loss design.
