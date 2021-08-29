import os

from torchdrug import data, utils
from torchdrug.core import Registry as R
from torchdrug.utils import doc


@R.register("datasets.ZINC250k")
@doc.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class ZINC250k(data.MoleculeDataset):
    """
    Subset of ZINC compound database for virtual screening.

    Statistics:
        - #Molecule: 498,910
        - #Regression task: 2

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/" \
          "250k_rndm_zinc_drugs_clean_3.csv"
    md5 = "b59078b2b04c6e9431280e3dc42048d5"
    target_fields = ["logP", "qed"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        if not os.path.exists(os.path.join(self.path, os.path.basename(self.url))):
            file_name = utils.download(self.url, self.path, md5=self.md5)
        else:
            file_name = os.path.join(self.path, os.path.basename(self.url))

        self.load_csv(file_name, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)