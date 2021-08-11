import os

from torchdrug import data, utils
from torchdrug.core import Registry as R
from torchdrug.utils import doc


@R.register("datasets.Tox21")
@doc.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class Tox21(data.MoleculeDataset):
    """
    Qualitative toxicity measurements on 12 biological targets, including nuclear receptors
    and stress response pathways.

    Statistics:
        - #Molecule: 7,831
        - #Classification task: 12

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    md5 = "2882d69e70bba0fec14995f26787cc25"
    target_fields = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
                     "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        csv_file = utils.extract(zip_file)

        self.load_csv(csv_file, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)