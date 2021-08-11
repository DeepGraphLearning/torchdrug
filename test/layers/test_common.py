import unittest

import torch
from torch import nn

from torchdrug import layers


class CommonTest(unittest.TestCase):

    def setUp(self):
        self.a = torch.randn(10)
        self.b = torch.randn(10)
        self.g = torch.randn(10)

    def test_sequential(self):
        layer1 = nn.Module()
        layer2 = nn.Module()
        layer3 = nn.Module()

        layer1.forward = lambda a, b: (a + 1, b + 2)
        layer2.forward = lambda a, b: a * b
        layer = layers.Sequential(layer1, layer2)
        result = layer(self.a, self.b)
        truth = layer2(*layer1(self.a, self.b))
        self.assertTrue(torch.allclose(result, truth), "Incorrect sequential layer")

        layer1.forward = lambda g, a: g + a
        layer2.forward = lambda b: b * 2
        layer3.forward = lambda g, c: g * c
        layer = layers.Sequential(layer1, layer2, layer3, global_args=("g",))
        result = layer(self.g, self.a)
        truth = layer3(self.g, layer2(layer1(self.g, self.a)))
        self.assertTrue(torch.allclose(result, truth), "Incorrect sequential layer")

        layer1.forward = lambda a: {"b": a + 1, "c": a + 2}
        layer2.forward = lambda b: b * 2
        layer = layers.Sequential(layer1, layer2, allow_unused=True)
        result = layer(self.a)
        truth = layer2(layer1(self.a)["b"])
        self.assertTrue(torch.allclose(result, truth), "Incorrect sequential layer")

        layer1.forward = lambda g, a: {"g": g + 1, "b": a + 2}
        layer2.forward = lambda g, b: g * b
        layer = layers.Sequential(layer1, layer2, global_args=("g",))
        result = layer(self.g, self.a)
        truth = layer2(**layer1(self.g, self.a))
        self.assertTrue(torch.allclose(result, truth), "Incorrect sequential layer")


if __name__ == "__main__":
    unittest.main()