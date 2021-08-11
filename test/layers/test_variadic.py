import unittest

import torch

from torchdrug.layers import functional


class VariadicTest(unittest.TestCase):

    def setUp(self):
        self.num_graph = 4
        self.size = torch.randint(3, 6, (self.num_graph,))
        self.num_node = self.size.sum()
        self.feature_dim = 2
        self.input = torch.rand(self.num_node, self.feature_dim)

    def test_variadic(self):
        for k in [self.size.min(), self.size.max()]:
            result_value, result_index = functional.variadic_topk(self.input, self.size, k)
            padded = torch.zeros(self.num_graph, self.size.max(), self.feature_dim)
            padded[:] = float("-inf")
            offset = 0
            for i, size in enumerate(self.size):
                padded[i, :size] = self.input[offset: offset + size]
                offset += size
            truth_value, truth_index = padded.topk(k, dim=1)
            for i, size in enumerate(self.size):
                for j in range(size, k):
                    truth_value[i, j] = truth_value[i, j-1]
                    truth_index[i, j] = truth_index[i, j-1]
            self.assertTrue(torch.equal(result_value, truth_value), "Incorrect variadic topk")
            self.assertTrue(torch.equal(result_index, truth_index), "Incorrect variadic topk")


if __name__ == "__main__":
    unittest.main()