import os

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.HIV")
@utils.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class HIV(data.MoleculeDataset):
    """
    Experimentally measured abilities to inhibit HIV replication.

    Statistics:
        - #Molecule: 41,127
        - #Classification task: 1

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/HIV.csv"
    md5 = "9ad10c88f82f1dac7eb5c52b668c30a7"
    target_fields = ["HIV_active"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        file_name = utils.download(self.url, path, md5=self.md5)

        self.load_csv(file_name, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)