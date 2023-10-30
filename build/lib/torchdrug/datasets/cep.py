import os

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.CEP")
@utils.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class CEP(data.MoleculeDataset):
    """
    Photovoltaic efficiency estimated by Havard clean energy project.

    Statistics:
        - #Molecule: 20,000
        - #Regression task: 1

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-02-cep-pce/cep-processed.csv"
    md5 = "b6d257ff416917e4e6baa5e1103f3929"
    target_fields = ["PCE"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        file_name = utils.download(self.url, self.path, md5=self.md5)

        self.load_csv(file_name, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)
