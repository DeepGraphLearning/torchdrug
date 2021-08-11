import unittest

import torch

from torchdrug import utils


class TorchExtensionTest(unittest.TestCase):

    def test_transfer(self):
        data = {"a": torch.rand(3), "b": [torch.rand(2), torch.rand(4)]}
        result = utils.cuda(data)
        truth = {"a": data["a"].cuda(), "b": [data["b"][0].cuda(), data["b"][1].cuda()]}
        self.assertTrue((result["a"] == truth["a"]).all(), "Incorrect CPU to GPU transfer")
        self.assertTrue((result["b"][0] == truth["b"][0]).all(), "Incorrect CPU to GPU transfer")
        self.assertTrue((result["b"][1] == truth["b"][1]).all(), "Incorrect CPU to GPU transfer")

        data, truth = truth, data
        result = utils.cpu(data)
        self.assertTrue((result["a"] == truth["a"]).all(), "Incorrect GPU to CPU transfer")
        self.assertTrue((result["b"][0] == truth["b"][0]).all(), "Incorrect GPU to CPU transfer")
        self.assertTrue((result["b"][1] == truth["b"][1]).all(), "Incorrect GPU to CPU transfer")

    def test_sparse_coo_tensor(self):
        num_node = 10
        num_edge = 40
        indices = torch.randint(0, num_node, (2, num_edge))
        values = torch.rand(num_edge)
        shape = (num_node, num_node)
        result = utils.sparse_coo_tensor(indices, values, shape)
        truth = torch.sparse_coo_tensor(indices, values, shape)
        self.assertTrue(torch.equal(result._indices(), truth._indices()), "Incorrect sparse COO tensor construction")
        self.assertTrue(torch.equal(result._values(), truth._values()), "Incorrect sparse COO tensor construction")
        self.assertEqual(result.shape, truth.shape, "Incorrect sparse COO tensor construction")


if __name__ == "__main__":
    unittest.main()