import os
from collections import defaultdict

from tqdm import tqdm
from rdkit import Chem, RDLogger

import torch

from torchdrug import data, utils
from torchdrug.data import feature
from torchdrug.core import Registry as R
from torchdrug.utils import doc


@R.register("datasets.QM9")
@doc.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class QM9(data.MoleculeDataset):
    """
    Geometric, energetic, electronic and thermodynamic properties of DFT-modeled small molecules.

    Statistics:
        - #Molecule: 133,885
        - #Regression task: 12

    Parameters:
        path (str): path to store the dataset
        node_position (bool, optional): load node position or not.
            This will add `node_position` as a node attribute to each sample.
        verbose (int, optional): output verbose level
        **kwargs
    """
    
    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
    md5 = "560f62d8e6c992ca0cf8ed8d013f9131"
    target_fields = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298"]

    def __init__(self, path, node_position=False, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        sdf_file = utils.extract(zip_file, "gdb9.sdf")
        csv_file = utils.extract(zip_file, "gdb9.sdf.csv")

        self.load_csv(csv_file, smiles_field=None, target_fields=self.target_fields, verbose=verbose)

        with utils.no_rdkit_log():
            molecules = Chem.SDMolSupplier(sdf_file, True, True, False)

        targets = self.targets
        self.data = []
        self.targets = defaultdict(list)
        assert len(molecules) == len(targets[self.target_fields[0]])
        indexes = range(len(molecules))
        if verbose:
            indexes = tqdm(indexes, "Constructing molecules from SDF")
        for i in indexes:
            with utils.capture_rdkit_log() as log:
                mol = molecules[i]
            if mol is None:
                continue
            if log.content:
                print(log.content)
            d = data.Molecule.from_molecule(mol, **kwargs)
            if node_position:
                with d.node():
                    d.node_position = torch.tensor([feature.atom_position(atom) for atom in mol.GetAtoms()])
            self.data.append(d)
            for k in targets:
                self.targets[k].append(targets[k][i])