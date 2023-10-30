import os

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.PCQM4M")
@utils.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class PCQM4M(data.MoleculeDataset):
    """
    Quantum chemistry dataset originally curated under the PubChemQC of molecules.

    Statistics:
        - #Molecule: 3,803,453
        - #Regression task: 1

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m_kddcup2021.zip"
    md5 = "5144ebaa7c67d24da1a2acbe41f57f6a"
    target_fields = ["homolumogap"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        
        zip_file = utils.download(self.url, self.path, md5=self.md5)
        zip_file = utils.extract(zip_file, "pcqm4m_kddcup2021/raw/data.csv.gz")
        file_name = utils.extract(zip_file)

        self.load_csv(file_name, smiles_field="smiles", target_fields=self.target_fields,
                      lazy=True, verbose=verbose, **kwargs)
