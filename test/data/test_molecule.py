import unittest

import torch

from torchdrug import data


class MoleculeTest(unittest.TestCase):

    def setUp(self):
        self.smiles = "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@](C#N)" \
                      "([C@H](O)[C@@H]1O)C1=CC=C2N1N=CN=C2N)OC1=CC=CC=C1"

    def test_smiles(self):
        mol = data.Molecule.from_smiles(self.smiles)
        smiles = mol.to_smiles().upper()
        carbon_result = (mol.atom_type == 6).sum().item()
        carbon_truth = self.smiles.count("C")
        atom_result = mol.num_atom
        atom_truth = self.smiles.count("C") + self.smiles.count("O") + self.smiles.count("N") + self.smiles.count("P")
        self.assertEqual(carbon_result, carbon_truth, "Incorrect SMILES construction")
        self.assertEqual(atom_result, atom_truth, "Incorrect SMILES construction")
        carbon_result = smiles.count("C")
        atom_result = smiles.count("C") + smiles.count("O") + smiles.count("N") + smiles.count("P")
        self.assertEqual(carbon_result, carbon_truth, "Incorrect SMILES construction")
        self.assertEqual(atom_result, atom_truth, "Incorrect SMILES construction")

        mol = data.Molecule.from_smiles("")
        self.assertEqual(mol.num_node, 0, "Incorrect SMILES side case")
        self.assertEqual(mol.num_edge, 0, "Incorrect SMILES side case")
        mols = data.PackedMolecule.from_smiles([""])
        self.assertTrue((mols.num_nodes == 0).all(), "Incorrect SMILES side case")
        self.assertTrue((mols.num_edges == 0).all(), "Incorrect SMILES side case")

    def test_feature(self):
        mol = data.Molecule.from_smiles(self.smiles, graph_feature="ecfp")
        self.assertTrue((mol.graph_feature > 0).any(), "Incorrect ECFP feature")

    def test_feature_with_kwargs(self):
        from torchdrug.core import Registry as R
        @R.register('features.atom.my_features')
        def my_features(atom, i, j):
            return [i, j]

        @R.register('features.bond.my_features')
        def my_features(bond, i, j):
            return [i, j]

        @R.register('features.molecule.my_features')
        def my_features(mol, i, j):
            return [i, j]

        expected_node_features = torch.tensor([1,2]).repeat((6,1))
        expected_edge_features = torch.tensor([1, 2]).repeat((12, 1))
        expected_graph_features = torch.tensor([1, 2])

        m = data.Molecule.from_smiles("C1=CC=CC=C1",
                                      node_feature="my_features",
                                      node_feature_kwargs=dict(i=1, j=2),
                                      edge_feature="my_features",
                                      edge_feature_kwargs=dict(i=1, j=2),
                                      graph_feature="my_features",
                                      graph_feature_kwargs=dict(i=1, j=2))

        assert (m.node_feature == expected_node_features).all()
        assert (m.edge_feature == expected_edge_features).all()
        assert (m.graph_feature == expected_graph_features).all()


if __name__ == "__main__":
    unittest.main()