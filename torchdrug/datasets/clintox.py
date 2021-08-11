import os

from torchdrug import data, utils
from torchdrug.core import Registry as R
from torchdrug.utils import doc


@R.register("datasets.ClinTox")
@doc.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class ClinTox(data.MoleculeDataset):
    """
    Qualitative data of drugs approved by the FDA and those that have failed clinical
    trials for toxicity reasons.

    Statistics:
        - #Molecule: 1,478
        - #Classification task: 2

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
    md5 = "db4f2df08be8ae92814e9d6a2d015284"
    target_fields = ["FDA_APPROVED", "CT_TOX"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        csv_file = utils.extract(zip_file)

        self.load_csv(csv_file, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)