# Lab Answers

## Q1. Why is the chain-rule decomposition useful for generating text?

For a sequence, the chain rule rewrites the joint probability as a product of next-token conditional probabilities:

`P(x1, ..., xT) = P(x1) * P(x2|x1) * ... * P(xT|x1,...,xT-1)`.

This is useful for generation because a complete sequence can be generated one token at a time. At each step, the model only needs to estimate the probability distribution of the next token given the previous context, sample one token, append it, and continue. The sequence probability is therefore broken into manageable conditional predictions.

## Q2. What independence assumption is made by the first-order network?

The first-order model assumes:

`P(X_t | X_1, ..., X_{t-1}) ≈ P(X_t | X_{t-1})`.

Equivalently, conditioned on the immediately previous token, earlier tokens are treated as irrelevant to predicting the current token.

## Q3. Construct `P(next word | current word)` and identify zero-probability transitions.

Training sentences:

1. the cat sat on the mat
2. the cat sat on the rug
3. the dog sat on the mat
4. the dog ran to the park
5. the cat ran to the park
6. the dog sat on the rug

With `<START>` and `<END>` added, the required distributions are:

| Current token | Next-token probabilities |
|---|---|
| `the` | cat: 0.25, dog: 0.25, mat: 1/6, rug: 1/6, park: 1/6 |
| `cat` | sat: 2/3, ran: 1/3 |
| `dog` | sat: 2/3, ran: 1/3 |
| `sat` | on: 1.0 |
| `ran` | to: 1.0 |

Other observed transitions are:

- `START -> the`: 1.0
- `on -> the`: 1.0
- `to -> the`: 1.0
- `mat -> END`: 1.0
- `rug -> END`: 1.0
- `park -> END`: 1.0

Any transition not observed from a given current token has probability 0 under this unsmoothed model. For example:

- `P(ran | the) = 0`
- `P(cat | cat) = 0`
- `P(dog | cat) = 0`
- `P(on | ran) = 0`

## Q4. Where are transition counts stored?

In `language_models.py`, the first-order model stores them in `FirstOrderLM.transition_counts`, a mapping from the current token to a `Counter` of observed next tokens.

Conceptually:

`transition_counts[current][next] += 1`

## Q5. Where is `P(X_t | X_{t-1})` computed?

The method `FirstOrderLM.probabilities_for()` converts the transition counts for a context into normalized probabilities by dividing each successor count by the total number of observed outgoing transitions.

Mathematically:

`P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k)`.

## Q6. How does the program choose the next word? Greedy or sampling?

The implementation supports both:

- **Greedy mode:** choose `argmax_w P(w | w_previous)`.
- **Sampling mode:** sample a token according to the full conditional distribution.

These are different behaviours. Greedy generation is deterministic for a fixed tie-breaking order and tends to follow the locally highest-probability transition. Sampling preserves the uncertainty represented by the model and therefore creates more variation.

## Q7. What happens if no transition has been observed?

The conditional distribution is empty because no training evidence exists for that context. This implementation handles that case explicitly by stopping generation rather than inventing an unsupported probability distribution.

A production language model could instead use smoothing, backoff, or an unknown-token strategy, but none of those are specified in this lab.

## Q8. If one normalization total is 0.87, what does it mean?

For a valid conditional probability distribution,

`sum_v P(v | w) = 1`.

Therefore a total of `0.87` means the implementation is incorrect. Probability mass is missing or the normalization step/counting logic is wrong.

## Q9. Are the most probable predictions always the words a person would expect?

No. The model only uses the conditional probabilities learned from the training data and its limited context. Human expectations can depend on broader semantics and context that this simple first-order model does not represent.

For example, a human can use a sentence-level interpretation, while the first-order model only sees the immediately previous token.

## Q10. Greedy vs sampling: which gives more variation and why?

Sampling produces more variation because it can select any token with non-zero probability according to the learned distribution.

Greedy generation always chooses the most probable local successor, so it is much more deterministic and can repeatedly follow the same path. In this dataset, first-order greedy generation can even become trapped in a loop such as `the -> cat -> sat -> on -> the ...` because the model lacks enough context to know that the second `the` should lead toward a sentence ending.

## Q11. How does the second-order model differ from the first-order model?

### 1. Graph structure

First-order:

`X_{t-1} -> X_t`

Second-order:

`X_{t-2} -> X_t <- X_{t-1}`

### 2. Conditional probability table

First-order stores `P(X_t | X_{t-1})`.

Second-order stores `P(X_t | X_{t-2}, X_{t-1})`.

### 3. Context

First-order uses one previous token. Second-order uses two previous tokens.

### 4. Data requirements

Second-order needs more observations because the number of possible contexts grows when two-token contexts are considered. Sparse data therefore creates more unseen contexts/zero probabilities.

## Q12. Why can more context improve prediction but make estimation harder?

More context can distinguish situations that look identical under a shorter context. Therefore it can make the conditional distribution more informative.

However, the number of possible contexts grows rapidly. With limited training data, many context combinations are never observed, leaving many zero or poorly estimated probabilities. The conditional probability table therefore becomes larger and sparser.

## Q13. Why is a behavioural specification preferable when asking an LLM to implement the model?

A behavioural specification states what probabilistic model the code must implement before code is generated. That makes the implementation testable against an explicit mathematical specification.

The important points are:

1. specify the intended behaviour;
2. understand the representation;
3. inspect generated code instead of trusting it blindly;
4. test probabilistic invariants such as normalization;
5. distinguish the probabilistic model from the code used to implement it.

Approach B is therefore preferable because the desired model is explicit: `P(X_t | X_{t-1})` estimated from transition counts with sampling-based generation.

## Q14. What did thinking of the language model as a Bayesian network add?

At least three useful things are visible:

1. **Dependency representation:** the graph makes clear which previous tokens influence the current token.
2. **Joint-distribution factorization:** the chain rule converts a joint probability into conditional factors.
3. **Conditional-probability interpretation:** each transition table entry has a direct probabilistic meaning.
4. **Principled generation:** generation becomes repeated sampling from conditional distributions.
5. **Independence assumptions:** the graph makes the first-order assumption explicit.
6. **Context effects:** moving from first-order to second-order clearly shows what additional context changes.
7. **Validation:** graph structure and probability invariants give concrete tests for whether the implementation matches the intended model.

## First-order model: worked normalization examples

For `the`:

`3 + 3 + 2 + 2 + 2 = 12` observed outgoing transitions.

Therefore:

- `P(cat | the) = 3/12 = 0.25`
- `P(dog | the) = 3/12 = 0.25`
- `P(mat | the) = 2/12 = 1/6`
- `P(rug | the) = 2/12 = 1/6`
- `P(park | the) = 2/12 = 1/6`

and the probabilities sum to 1.

For `cat`:

`2 + 1 = 3`, so:

- `P(sat | cat) = 2/3`
- `P(ran | cat) = 1/3`

## Second-order non-zero contexts from the supplied dataset

| Context | Next-token probabilities |
|---|---|
| (`START`, `the`) | cat: 0.5, dog: 0.5 |
| (`the`, `cat`) | sat: 2/3, ran: 1/3 |
| (`the`, `dog`) | sat: 2/3, ran: 1/3 |
| (`cat`, `sat`) | on: 1.0 |
| (`dog`, `sat`) | on: 1.0 |
| (`cat`, `ran`) | to: 1.0 |
| (`dog`, `ran`) | to: 1.0 |
| (`sat`, `on`) | the: 1.0 |
| (`on`, `the`) | mat: 0.5, rug: 0.5 |
| (`ran`, `to`) | the: 1.0 |
| (`to`, `the`) | park: 1.0 |
| (`the`, `mat`) | END: 1.0 |
| (`the`, `rug`) | END: 1.0 |
| (`the`, `park`) | END: 1.0 |

## Model comparison on this dataset

### Number of non-zero stored transition probabilities

- First-order: 17
- Second-order: 18

The exact number of zero-probability contexts depends on how the context universe is defined. The second-order model has a much larger possible context space, so with only six training sentences it is substantially sparser.

### Expected qualitative behaviour

- First-order model: more ambiguity and less coherent context because it sees only one previous word.
- Second-order model: better local coherence because two previous tokens distinguish contexts such as `the cat` and `the dog`.
- Greedy first-order can loop because `the` does not reveal enough context.
- Second-order greedy can produce a complete sentence such as `the cat sat on the mat`.
- Sampling produces more variation than greedy generation.

## Reflection draft

I used an LLM as an implementation assistant, but treated the probabilistic specification as the source of truth. I first specified the intended model as `P(X_t | X_{t-1})` for the first-order case and `P(X_t | X_{t-2}, X_{t-1})` for the second-order case. I then inspected the generated implementation to verify where counts were stored, where normalization was performed, and how the next token was selected.

One important validation step was checking the invariant `sum_v P(v | w) = 1` for every observed context. I also compared greedy generation with sampling and tested unseen-context handling instead of allowing the implementation to invent probabilities. The key lesson was that an LLM can produce code quickly, but the probabilistic model must be specified independently and the implementation must be validated against that specification.
