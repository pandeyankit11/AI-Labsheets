import unittest

import torch

from xor_lab import (
    X_XOR,
    Y_XOR,
    activation_experiments,
    train_linear_baseline,
    train_symmetry_experiment,
    train_three_class,
)


class NeuralModelsLabTests(unittest.TestCase):
    def test_linear_model_cannot_solve_xor(self):
        result = train_linear_baseline()
        self.assertLess(result["correct"], 4)

    def test_xor_activations_learn(self):
        for result in activation_experiments():
            self.assertEqual(result.predictions, [0, 1, 1, 0], msg=result.activation)
            self.assertLess(result.final_loss, 0.01, msg=result.activation)

    def test_first_layer_gradient_tensor_is_recorded(self):
        for result in activation_experiments():
            gradient = torch.tensor(result.early_first_layer_gradient)
            self.assertEqual(tuple(gradient.shape), (2, 2))
            self.assertGreater(float(gradient.abs().sum().item()), 0.0)
            self.assertAlmostEqual(
                float(gradient.norm().item()),
                result.early_first_layer_grad_norm,
                places=6,
            )

    def test_zero_initialisation_preserves_hidden_symmetry(self):
        result = train_symmetry_experiment()
        self.assertTrue(all(row["row_difference"] == 0.0 for row in result["rows"]))
        self.assertTrue(all(row["gradient_norm"] == 0.0 for row in result["rows"]))

    def test_three_class_probabilities_normalise(self):
        result = train_three_class()
        self.assertEqual(result["predictions"], [0, 1, 1, 2])
        self.assertTrue(all(abs(total - 1.0) < 1e-6 for total in result["probability_sums"]))
        self.assertLess(result["max_shift_invariance_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
