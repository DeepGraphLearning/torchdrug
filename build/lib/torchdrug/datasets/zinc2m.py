import os
import csv
from tqdm import tqdm
import shutil

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.ZINC2m")
@utils.copy_args(data.MoleculeDataset.load_smiles, ignore=("smiles_field", "target_fields"))
class ZINC2m(data.MoleculeDataset):
    """
    ZINC compound database for virtual screening.
    This dataset doesn't contain any label information.

    Statistics:
        - #Molecule: 2,000,000

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    target_fields = []

    url = "http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip"
    md5 = "e95da4dffa0fdb1d4af2726bdf8c23e0"
    member = "dataset/zinc_standard_agent/processed/smiles.csv"

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        zip_file_name = utils.download(self.url, path, md5=self.md5)

        save_file = utils.extract(zip_file=zip_file_name, member=self.member)
        neo_save_file = os.path.join(os.path.dirname(zip_file_name), 'zinc2m_'+os.path.basename(self.member))
        shutil.move(save_file, neo_save_file)

        with open(neo_save_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" % path, utils.get_line_count(neo_save_file)))
            smiles_list = []

            for idx, values in enumerate(reader):
                smiles = values[0]
                smiles_list.append(smiles)

        targets = {}
        self.load_smiles(smiles_list, targets, lazy=True, verbose=verbose, **kwargs)
