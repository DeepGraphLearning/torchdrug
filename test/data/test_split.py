import unittest

import torch

from torchdrug import data, datasets


class SplitTest(unittest.TestCase):

    def setUp(self):
        smiles = ["CC1CCC(C(C)C)C(O)C1", # scaffold: C1CCCCC1
                  "OC1CCCCC1",
                  "CCSC(=O)N(CC)C1CCCCC1",
                  "ClC1C(Cl)C(Cl)C(Cl)C(Cl)C1Cl",
                  "CC1CCC(C)CC1",
                  "CCN(CC)c1nc(Cl)nc(N(CC)CC)n1", # scaffold: c1ncncn1
                  "CCNc1nc(NC(C)C)nc(SC)n1",
                  "CCNc1nc(NC(C)(C)C)nc(SC)n1",
                  "CCNc1nc(NC(C)C)nc(OC)n1",
                  "CCNc1nc(Cl)nc(NCC)n1"]
        self.dataset = data.MoleculeDataset()
        self.dataset.load_smiles(smiles, {})
        self.lengths = [5, 5]

    def test_scaffold(self):
        train, test = data.scaffold_split(self.dataset, self.lengths)
        train_scaffolds = set(sample["graph"].to_scaffold() for sample in train)
        test_scaffolds = set(sample["graph"].to_scaffold() for sample in test)
        self.assertEqual(len(train_scaffolds), 1, "Incorrect scaffold split")
        self.assertEqual(len(test_scaffolds), 1, "Incorrect scaffold split")
        self.assertFalse(train_scaffolds.intersection(test_scaffolds), "Incorrect scaffold split")


if __name__ == "__main__":
    unittest.main()