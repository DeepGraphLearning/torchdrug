import os
from collections import defaultdict

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R
from torchdrug.utils import doc


@R.register("datasets.MOSES")
@doc.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class MOSES(data.MoleculeDataset):
    """
    Subset of ZINC database for molecule generation.
    This dataset doesn't contain any label information.

    Statistics:
        - #Molecule: 1,936,963

    Parameters:
        path (str): path for the CSV dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
    md5 = "6bdb0d9526ddf5fdeb87d6aa541df213"
    target_fields = ["SPLIT"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        file_name = utils.download(self.url, path, md5=self.md5)

        self.load_csv(file_name, smiles_field="SMILES", target_fields=self.target_fields,
                      lazy=True, verbose=verbose, **kwargs)

    def split(self):
        indexes = defaultdict(list)
        for i, split in enumerate(self.targets["SPLIT"]):
            indexes[split].append(i)
        train_set = torch_data.Subset(self, indexes["train"])
        valid_set = torch_data.Subset(self, indexes["valid"])
        test_set = torch_data.Subset(self, indexes["test"])
        return train_set, valid_set, test_set
