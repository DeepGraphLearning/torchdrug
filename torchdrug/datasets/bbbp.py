import os

from torchdrug import data, utils
from torchdrug.utils import doc
from torchdrug.core import Registry as R


@R.register("datasets.BBBP")
@doc.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class BBBP(data.MoleculeDataset):
    """
    Binary labels of blood-brain barrier penetration.

    Statistics:
        - #Molecule: 2,039
        - #Classification task: 1

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/BBBP.csv"
    md5 = "66286cb9e6b148bd75d80c870df580fb"
    target_fields = ["p_np"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        file_name = utils.download(self.url, path, md5=self.md5)

        self.load_csv(file_name, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)