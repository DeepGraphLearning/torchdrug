import os

from torchdrug import data, utils
from torchdrug.core import Registry as R
from torchdrug.utils import doc


@R.register("datasets.Malaria")
@doc.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class Malaria(data.MoleculeDataset):
    """
    Half-maximal effective concentration (EC50) against a parasite that causes malaria.

    Statistics:
        - #Molecule: 10,000
        - #Regression task: 1

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-03-malaria/" \
          "malaria-processed.csv"
    md5 = "ef40ddfd164be0e5ed1bd3dd0cce9b88"
    target_fields = ["activity"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        file_name = utils.download(self.url, self.path, md5=self.md5)

        self.load_csv(file_name, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)