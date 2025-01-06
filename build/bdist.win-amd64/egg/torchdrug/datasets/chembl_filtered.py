import os

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.ChEMBLFiltered")
@utils.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class ChEMBLFiltered(data.MoleculeDataset):
    """
    Statistics:
        - #Molecule: 430,710
        - #Regression task: 1,310

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://zenodo.org/record/5528681/files/chembl_filtered_torchdrug.csv.gz"
    md5 = "2fff04fecd6e697f28ebb127e8a37561"

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        csv_file = utils.extract(zip_file)

        self.target_fields = ["target_{}".format(i) for i in range(1310)]

        self.load_csv(csv_file, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)