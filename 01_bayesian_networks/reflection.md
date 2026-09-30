# Reflection

I used an LLM as an implementation assistant, but treated the mathematical/probabilistic specification as the source of truth.

I first specified the intended model as `P(X_t | X_{t-1})` for the first-order case and `P(X_t | X_{t-2}, X_{t-1})` for the second-order case. I then inspected the implementation to verify where transition counts were stored, where probabilities were normalized, how greedy prediction differed from sampling, and how an unseen context was handled.

The main validation was the probability invariant `sum_v P(v | w) = 1` for every observed first-order context and the corresponding check for each observed second-order context. I also compared greedy and sampling generation and verified that the generated behaviour followed the specified probabilistic structure.

The key lesson is that code generation and model specification are different tasks: the LLM can help implement a model, but the intended probability model must be understood and the generated implementation must be tested against explicit invariants.
