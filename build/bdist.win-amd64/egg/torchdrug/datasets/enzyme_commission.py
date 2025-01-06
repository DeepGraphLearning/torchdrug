import os
import csv
import glob

import torch
from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.EnzymeCommission")
@utils.copy_args(data.ProteinDataset.load_pdbs)
class EnzymeCommission(data.ProteinDataset):
    """
    A set of proteins with their 3D structures and EC numbers, which describes their
    catalysis of biochemical reactions.

    Statistics (test_cutoff=0.95):
        - #Train: 15,011
        - #Valid: 1,664
        - #Test: 1,840

    Parameters:
        path (str): the path to store the dataset
        test_cutoff (float, optional): the test cutoff used to split the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://zenodo.org/record/6622158/files/EnzymeCommission.zip"
    md5 = "33f799065f8ad75f87b709a87293bc65"
    processed_file = "enzyme_commission.pkl.gz"
    test_cutoffs = [0.3, 0.4, 0.5, 0.7, 0.95]

    def __init__(self, path, test_cutoff=0.95, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        if test_cutoff not in self.test_cutoffs:
            raise ValueError("Unknown test cutoff `%.2f` for EnzymeCommission dataset" % test_cutoff)
        self.test_cutoff = test_cutoff

        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(utils.extract(zip_file), "EnzymeCommission")
        pkl_file = os.path.join(path, self.processed_file)

        csv_file = os.path.join(path, "nrPDB-EC_test.csv")
        pdb_ids = []
        with open(csv_file, "r") as fin:
            reader = csv.reader(fin, delimiter=",")
            idx = self.test_cutoffs.index(test_cutoff) + 1
            _ = next(reader)
            for line in reader:
                if line[idx] == "0":
                    pdb_ids.append(line[0])

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = []
            for split in ["train", "valid", "test"]:
                split_path = utils.extract(os.path.join(path, "%s.zip" % split))
                pdb_files += sorted(glob.glob(os.path.join(split_path, split, "*.pdb")))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)
        if len(pdb_ids) > 0:
            self.filter_pdb(pdb_ids)

        tsv_file = os.path.join(path, "nrPDB-EC_annot.tsv")
        pdb_ids = [os.path.basename(pdb_file).split("_")[0] for pdb_file in self.pdb_files]
        self.load_annotation(tsv_file, pdb_ids)

        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [splits.count("train"), splits.count("valid"), splits.count("test")]

    def filter_pdb(self, pdb_ids):
        pdb_ids = set(pdb_ids)
        sequences = []
        pdb_files = []
        data = []
        for sequence, pdb_file, protein in zip(self.sequences, self.pdb_files, self.data):
            if os.path.basename(pdb_file).split("_")[0] in pdb_ids:
                continue
            sequences.append(sequence)
            pdb_files.append(pdb_file)
            data.append(protein)
        self.sequences = sequences
        self.pdb_files = pdb_files
        self.data = data

    def load_annotation(self, tsv_file, pdb_ids):
        with open(tsv_file, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            _ = next(reader)
            tasks = next(reader)
            task2id = {task: i for i, task in enumerate(tasks)}
            _ = next(reader)
            pos_targets = {}
            for pdb_id, pos_target in reader:
                pos_target = [task2id[t] for t in pos_target.split(",")]
                pos_target = torch.tensor(pos_target)
                pos_targets[pdb_id] = pos_target

        # fake targets to enable the property self.tasks
        self.targets = task2id
        self.pos_targets = []
        for pdb_id in pdb_ids:
            self.pos_targets.append(pos_targets[pdb_id])

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

    def get_item(self, index):
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        indices = self.pos_targets[index].unsqueeze(0)
        values = torch.ones(len(self.pos_targets[index]))
        item["targets"] = utils.sparse_coo_tensor(indices, values, (len(self.tasks),)).to_dense()
        return item