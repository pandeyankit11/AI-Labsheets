from pathlib import Path
from random import Random

from language_models import END, START, train_models

DATASET = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def print_first_order_tables(model) -> None:
    print("\nFIRST-ORDER CONDITIONAL PROBABILITIES")
    print("=" * 48)
    for word in ["the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"]:
        print(f"P(next | {word}) = {model.probabilities_for(word)}")


def print_second_order_tables(model) -> None:
    print("\nSECOND-ORDER CONDITIONAL PROBABILITIES")
    print("=" * 48)
    for context in sorted(model.trigram_counts):
        print(f"P(next | {context}) = {model.probabilities_for(context)}")


def save_generated_sentences(first, second, output_file: Path) -> None:
    rng_first = Random(42)
    rng_second = Random(42)

    with output_file.open("w", encoding="utf-8") as handle:
        handle.write("FIRST-ORDER SAMPLING (20 sentences, seed=42)\n")
        handle.write("-" * 60 + "\n")
        for number in range(1, 21):
            handle.write(f"{number:02d}. {first.generate(rng_first)}\n")

        handle.write("\nSECOND-ORDER SAMPLING (20 sentences, seed=42)\n")
        handle.write("-" * 60 + "\n")
        for number in range(1, 21):
            handle.write(f"{number:02d}. {second.generate(rng_second)}\n")

        handle.write("\nFIRST-ORDER GREEDY (5 runs; deterministic; max_tokens=12)\n")
        handle.write("-" * 60 + "\n")
        for number in range(1, 6):
            handle.write(
                f"{number:02d}. {first.generate(Random(number), greedy=True, max_tokens=12)}\n"
            )

        handle.write("\nSECOND-ORDER GREEDY (5 runs; deterministic)\n")
        handle.write("-" * 60 + "\n")
        for number in range(1, 6):
            handle.write(
                f"{number:02d}. {second.generate(Random(number), greedy=True, max_tokens=30)}\n"
            )
        handle.write(
            "\nNOTE: Greedy generation is deterministic, so repeated runs with the same "
            "model and tie-breaking order are expected to be identical. First-order "
            "greedy loops on this corpus and is capped at 12 tokens.\n"
        )


def main() -> None:
    first, second = train_models(DATASET)

    print_first_order_tables(first)
    print_second_order_tables(second)

    print("\nNORMALISATION CHECKS")
    print("=" * 48)
    print("First-order errors:", first.normalisation_errors())
    print("Second-order errors:", second.normalisation_errors())

    print("\nMODEL SIZE")
    print("=" * 48)
    print("First-order nonzero conditional-probability entries:", first.nonzero_parameters)
    print("Second-order nonzero conditional-probability entries (including initial distribution):", second.nonzero_parameters)
    print("  Trigram successor entries:", sum(len(counter) for counter in second.trigram_counts.values()))
    print("  Initial-token distribution entries:", len(second.first_token_counts))

    context_stats = second.context_statistics()
    print("\nSECOND-ORDER CONTEXT SPARSITY")
    print("Context universe: (START or lexical token, followed by lexical token)")
    print("Possible contexts:", context_stats["possible_contexts"])
    print("Observed contexts:", context_stats["observed_contexts"])
    print("Unobserved / zero-probability contexts:", context_stats["zero_probability_contexts"])

    print("\nNEXT-WORD PREDICTIONS")
    print("=" * 48)
    for word in ["the", "cat", "dog", "sat", "ran"]:
        print(f"argmax P(next | {word}) = {first.predict_next(word)}")

    print("\nGENERATED SENTENCES")
    print("=" * 48)
    output_file = Path("generated_examples.txt")
    save_generated_sentences(first, second, output_file)
    print(f"Saved examples to {output_file.resolve()}")


if __name__ == "__main__":
    main()
