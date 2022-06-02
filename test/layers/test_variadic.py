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
        self.padded = torch.zeros(self.num_graph, self.size.max(), self.feature_dim)
        self.padded[:] = float("-inf")
        offset = 0
        for i, size in enumerate(self.size):
            self.padded[i, :size] = self.input[offset: offset + size]
            offset += size

    def test_arange(self):
        result = functional.variadic_arange(self.size)
        truth = torch.cat([torch.arange(x) for x in self.size])
        self.assertTrue(torch.equal(result, truth), "Incorrect variadic arange")

    def test_sort(self):
        result_value, result_index = functional.variadic_sort(self.input, self.size, descending=True)
        truth_value, truth_index = self.padded.sort(dim=1, descending=True)
        mask = ~torch.isinf(self.padded)
        truth_value = truth_value[mask].view(-1, self.feature_dim)
        truth_index = truth_index[mask].view(-1, self.feature_dim)
        self.assertTrue(torch.equal(result_value, truth_value), "Incorrect variadic sort")
        self.assertTrue(torch.equal(result_index, truth_index), "Incorrect variadic sort")

    def test_topk(self):
        for k in [self.size.min(), self.size.max()]:
            result_value, result_index = functional.variadic_topk(self.input, self.size, k)
            truth_value, truth_index = self.padded.topk(k, dim=1)
            for i, size in enumerate(self.size):
                for j in range(size, k):
                    truth_value[i, j] = truth_value[i, j-1]
                    truth_index[i, j] = truth_index[i, j-1]
            self.assertTrue(torch.equal(result_value, truth_value), "Incorrect variadic topk")
            self.assertTrue(torch.equal(result_index, truth_index), "Incorrect variadic topk")

        for _ in range(10):
            k = torch.randint(self.size.min(), self.size.max(), (self.num_graph,))
            result_value, result_index = functional.variadic_topk(self.input, self.size, k)
            _truth_value, _truth_index = self.padded.topk(self.size.max(), dim=1)
            truth_value, truth_index = [], []
            for i, size in enumerate(self.size):
                truth_value.append(_truth_value[i, :k[i]])
                truth_index.append(_truth_index[i, :k[i]])
                for j in range(size, k[i].item()):
                    truth_value[i][j] = truth_value[i][j-1]
                    truth_index[i][j] = truth_index[i][j-1]
            truth_value = torch.cat(truth_value, dim=0)
            truth_index = torch.cat(truth_index, dim=0)
            self.assertTrue(torch.equal(result_value, truth_value), "Incorrect variadic topk")
            self.assertTrue(torch.equal(result_index, truth_index), "Incorrect variadic topk")

if __name__ == "__main__":
    unittest.main()