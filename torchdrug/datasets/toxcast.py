import os

from torchdrug import data, utils
from torchdrug.core import Registry as R
from torchdrug.utils import doc


@R.register("datasets.ToxCast")
@doc.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class ToxCast(data.MoleculeDataset):
    """
    Toxicology data based on in vitro high-throughput screening.

    Statistics:
        - #Molecule: 8,575
        - #Classification task: 617

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz"
    md5 = "92911bbf9c1e2ad85231014859388cd6"
    target_fields = None  # pick all targets

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        csv_file = utils.extract(zip_file)

        self.load_csv(csv_file, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)