import unittest

from torchdrug import models


class ConfigTest(unittest.TestCase):

    def setUp(self):
        self.input_dim = 64
        self.hidden_dims = [64, 64]
        self.num_mlp_layer = 3

    def test_config(self):
        model = models.GIN(self.input_dim, self.hidden_dims)
        infograph = models.InfoGraph(model, self.num_mlp_layer)
        config = infograph.config_dict()
        infograph2 = models.InfoGraph.load_config_dict(config)
        config2 = infograph2.config_dict()
        self.assertEqual(config, config2, "Incorrect load config dict")


if __name__ == "__main__":
    unittest.main()