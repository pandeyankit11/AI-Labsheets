from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from random import Random
from typing import DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

START = "<START>"
END = "<END>"


def prepare_sentences(raw_sentences: Iterable[str]) -> List[List[str]]:
    """Lowercase whitespace-tokenized sentences and add boundary markers."""
    prepared: List[List[str]] = []
    for sentence in raw_sentences:
        tokens = sentence.strip().lower().split()
        if not tokens:
            continue
        prepared.append([START, *tokens, END])
    if not prepared:
        raise ValueError("Training data must contain at least one non-empty sentence.")
    return prepared


def _normalise(counter: Mapping[str, int]) -> Dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {token: count / total for token, count in counter.items()}


def _sample(distribution: Mapping[str, float], rng: Random) -> str:
    if not distribution:
        raise ValueError("Cannot sample from an empty probability distribution.")

    threshold = rng.random()
    cumulative = 0.0
    last_token = None
    for token, probability in distribution.items():
        cumulative += probability
        last_token = token
        if threshold <= cumulative:
            return token
    # Protect against tiny floating-point accumulation error.
    assert last_token is not None
    return last_token


@dataclass
class FirstOrderLM:
    transition_counts: DefaultDict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def fit(self, tokenised_sentences: Sequence[Sequence[str]]) -> None:
        self.transition_counts.clear()
        for tokens in tokenised_sentences:
            if len(tokens) < 2:
                continue
            for current, nxt in zip(tokens, tokens[1:]):
                self.transition_counts[current][nxt] += 1
        if not self.transition_counts:
            raise ValueError("No transitions were found in the training data.")

    def counts_for(self, current: str) -> Dict[str, int]:
        return dict(self.transition_counts.get(current, {}))

    def probabilities_for(self, current: str) -> Dict[str, float]:
        return _normalise(self.transition_counts.get(current, {}))

    def predict_next(self, current: str) -> str | None:
        distribution = self.probabilities_for(current)
        if not distribution:
            return None
        return max(distribution.items(), key=lambda pair: pair[1])[0]

    def generate(
        self,
        rng: Random,
        *,
        greedy: bool = False,
        max_tokens: int = 30,
    ) -> str:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        current = START
        generated: List[str] = []
        for _ in range(max_tokens):
            distribution = self.probabilities_for(current)
            if not distribution:
                break
            next_token = (
                self.predict_next(current)
                if greedy
                else _sample(distribution, rng)
            )
            if next_token is None or next_token == END:
                break
            generated.append(next_token)
            current = next_token
        return " ".join(generated)

    def normalisation_errors(self, tolerance: float = 1e-12) -> Dict[str, float]:
        errors: Dict[str, float] = {}
        for current in self.transition_counts:
            total = sum(self.probabilities_for(current).values())
            if abs(total - 1.0) > tolerance:
                errors[current] = total
        return errors

    @property
    def nonzero_parameters(self) -> int:
        return sum(len(counter) for counter in self.transition_counts.values())


@dataclass
class SecondOrderLM:
    trigram_counts: DefaultDict[Tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    first_token_counts: Counter[str] = field(default_factory=Counter)

    def fit(self, tokenised_sentences: Sequence[Sequence[str]]) -> None:
        self.trigram_counts.clear()
        self.first_token_counts.clear()

        for tokens in tokenised_sentences:
            if len(tokens) < 2:
                continue
            # First token after START is handled explicitly during generation.
            self.first_token_counts[tokens[1]] += 1

            for index in range(2, len(tokens)):
                context = (tokens[index - 2], tokens[index - 1])
                self.trigram_counts[context][tokens[index]] += 1

        if not self.first_token_counts or not self.trigram_counts:
            raise ValueError("Insufficient data for a second-order model.")

    def probabilities_for(self, previous_two: Tuple[str, str]) -> Dict[str, float]:
        return _normalise(self.trigram_counts.get(previous_two, {}))

    def predict_next(self, previous_two: Tuple[str, str]) -> str | None:
        distribution = self.probabilities_for(previous_two)
        if not distribution:
            return None
        return max(distribution.items(), key=lambda pair: pair[1])[0]

    def first_token_distribution(self) -> Dict[str, float]:
        return _normalise(self.first_token_counts)

    def generate(
        self,
        rng: Random,
        *,
        greedy: bool = False,
        max_tokens: int = 30,
    ) -> str:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        first_distribution = self.first_token_distribution()
        first = (
            max(first_distribution.items(), key=lambda pair: pair[1])[0]
            if greedy
            else _sample(first_distribution, rng)
        )
        if first == END:
            return ""

        generated = [first]
        previous_two = (START, first)

        for _ in range(max_tokens - 1):
            distribution = self.probabilities_for(previous_two)
            if not distribution:
                break
            next_token = (
                self.predict_next(previous_two)
                if greedy
                else _sample(distribution, rng)
            )
            if next_token is None or next_token == END:
                break
            generated.append(next_token)
            previous_two = (previous_two[1], next_token)

        return " ".join(generated)

    def normalisation_errors(self, tolerance: float = 1e-12) -> Dict[Tuple[str, str], float]:
        errors: Dict[Tuple[str, str], float] = {}
        for context in self.trigram_counts:
            total = sum(self.probabilities_for(context).values())
            if abs(total - 1.0) > tolerance:
                errors[context] = total
        return errors

    @property
    def nonzero_parameters(self) -> int:
        return sum(len(counter) for counter in self.trigram_counts.values())


def train_models(raw_sentences: Iterable[str]) -> Tuple[FirstOrderLM, SecondOrderLM]:
    tokenised = prepare_sentences(raw_sentences)

    first = FirstOrderLM()
    first.fit(tokenised)

    second = SecondOrderLM()
    second.fit(tokenised)
    return first, second
