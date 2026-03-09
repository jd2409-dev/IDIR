
import torch
import unittest
from idir_model import IDIR

class TestIDIRModel(unittest.TestCase):

    def setUp(self):
        """Set up a model instance before each test."""
        self.vocab_size = 50000
        self.hidden_dim = 512
        self.model = IDIR(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_experts=4,
            expert_hidden_dim=2048,
            reasoning_steps=3,
            k_features=32,
            max_iterations=20,
            tolerance=1e-3
        )

    def test_model_instantiation(self):
        """Test that the model can be instantiated without errors."""
        self.assertIsInstance(self.model, IDIR)

    def test_forward_pass(self):
        """Test the forward pass with a dummy input tensor."""
        batch_size = 2
        seq_len = 16
        input_tensor = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        
        logits, iterations = self.model(input_tensor)
        
        # Check output shape
        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))
        
        # Check that iterations is a non-negative integer
        self.assertIsInstance(iterations, int)
        self.assertGreaterEqual(iterations, 0)

    def test_parameter_count(self):
        """Test the parameter count method."""
        param_counts = self.model.get_parameter_count()
        
        self.assertIn("Total", param_counts)
        self.assertGreater(param_counts["Total"], 0)

        # Verification from paper (approximate)
        # Embedding: 25,600,000
        # Output: 25,600,000
        # Implicit core: 524,288
        # Experts: 8,388,608
        # Routing: 2,048
        # Reasoning: ~6,000,000
        # Total: ~70,114,944

        self.assertEqual(param_counts["Embedding"], 25600000)
        self.assertEqual(param_counts["Output"], 25600000)
        self.assertEqual(param_counts["Implicit core"], 524288)
        
        # Note: My calculation for experts and routing may differ slightly 
        # based on whether biases are included. The paper is not explicit.
        # This is a sanity check.
        self.assertAlmostEqual(param_counts["Experts"], 8388608, delta=10000)
        self.assertAlmostEqual(param_counts["Routing"], 2048, delta=100)
        self.assertAlmostEqual(param_counts["Total"], 70114944, delta=6000000)


if __name__ == '__main__':
    unittest.main()
