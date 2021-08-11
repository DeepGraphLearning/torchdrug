import os
import csv
import math
from collections import defaultdict
from tqdm import tqdm

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R
from torchdrug.utils import doc


@R.register("datasets.OPV")
@doc.copy_args(data.MoleculeDataset.load_smiles)
class OPV(data.MoleculeDataset):
    """
    Quantum mechanical calculations on organic photovoltaic candidate molecules.

    Statistics:
        - #Molecule: 94,576
        - #Regression task: 8

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    train_url = "https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e433d5750d6/download/" \
                "b69cf9a5-e7e0-405b-88cb-40df8007242e"
    valid_url = "https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e433d5750d6/download/" \
                "1c8e7379-3071-4360-ba8e-0c6481c33d2c"
    test_url = "https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e433d5750d6/download/" \
               "4ef40592-0080-4f00-9bb7-34b25f94962a"
    train_md5 = "16e439b7411ea0a8d3a56ba4802b61b1"
    valid_md5 = "3aa2ac62015932ca84661feb5d29adda"
    test_md5 = "bad072224f0755478f0729476ca99a33"
    target_fields = ["gap", "homo", "lumo", "spectral_overlap", "gap_extrapolated", "homo_extrapolated",
                     "lumo_extrapolated", "optical_lumo_extrapolated"]

    def read_csv(self, csv_file, smiles_field="smiles", target_fields=None, verbose=0):
        if target_fields is not None:
            target_fields = set(target_fields)

        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
            fields = next(reader)
            smiles = []
            targets = defaultdict(list)
            for i, values in enumerate(reader):
                if not any(values):
                    continue
                if smiles_field is None:
                    smiles.append("")
                for field, value in zip(fields, values):
                    if field == smiles_field:
                        smiles.append(value)
                    elif target_fields is None or field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = math.nan
                        targets[field].append(value)

        return smiles, targets

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        train_zip_file = utils.download(self.train_url, path, save_file="mol_train.csv.gz", md5=self.train_md5)
        valid_zip_file = utils.download(self.valid_url, path, save_file="mol_valid.csv.gz", md5=self.valid_md5)
        test_zip_file = utils.download(self.test_url, path, save_file="mol_test.csv.gz", md5=self.test_md5)
        train_file = utils.extract(train_zip_file)
        valid_file = utils.extract(valid_zip_file)
        test_file = utils.extract(test_zip_file)

        train_smiles, train_targets = self.read_csv(train_file, smiles_field="smile", target_fields=self.target_fields)
        valid_smiles, valid_targets = self.read_csv(valid_file, smiles_field="smile", target_fields=self.target_fields)
        test_smiles, test_targets = self.read_csv(test_file, smiles_field="smile", target_fields=self.target_fields)
        self.num_train = len(train_smiles)
        self.num_valid = len(valid_smiles)
        self.num_test = len(test_smiles)

        smiles = train_smiles + valid_smiles + test_smiles
        targets = {k: train_targets[k] + valid_targets[k] + test_targets[k] for k in train_targets}

        self.load_smiles(smiles, targets, verbose=verbose, **kwargs)

    def split(self):
        train_set = torch_data.Subset(self, range(self.num_train))
        valid_set = torch_data.Subset(self, range(self.num_train, self.num_train + self.num_valid))
        test_set = torch_data.Subset(self, range(-self.num_test, 0))
        return train_set, valid_set, test_set