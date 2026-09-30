import unittest
from random import Random

from language_models import END, START, FirstOrderLM, SecondOrderLM, prepare_sentences

RAW = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


class LanguageModelTests(unittest.TestCase):
    def setUp(self):
        tokenised = prepare_sentences(RAW)
        self.first = FirstOrderLM()
        self.first.fit(tokenised)
        self.second = SecondOrderLM()
        self.second.fit(tokenised)

    def test_first_order_probabilities(self):
        self.assertAlmostEqual(self.first.probabilities_for("cat")["sat"], 2 / 3)
        self.assertAlmostEqual(self.first.probabilities_for("cat")["ran"], 1 / 3)
        self.assertAlmostEqual(self.first.probabilities_for("the")["cat"], 0.25)
        self.assertEqual(self.first.normalisation_errors(), {})

    def test_second_order_probabilities(self):
        probs = self.second.probabilities_for(("the", "cat"))
        self.assertAlmostEqual(probs["sat"], 2 / 3)
        self.assertAlmostEqual(probs["ran"], 1 / 3)
        self.assertEqual(self.second.normalisation_errors(), {})

    def test_greedy_generation(self):
        first_sentence = self.first.generate(Random(0), greedy=True, max_tokens=5)
        second_sentence = self.second.generate(Random(0), greedy=True, max_tokens=10)
        self.assertEqual(first_sentence, "the cat sat on the")
        self.assertEqual(second_sentence, "the cat sat on the mat")

    def test_unseen_context_is_explicit(self):
        self.assertEqual(self.first.probabilities_for("never-seen"), {})
        self.assertEqual(self.second.probabilities_for(("never", "seen")), {})


if __name__ == "__main__":
    unittest.main()
