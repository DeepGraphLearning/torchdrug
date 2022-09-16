import os

import torch
from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.SecondaryStructure")
@utils.copy_args(data.ProteinDataset.load_lmdbs, ignore=("target_fields",))
class SecondaryStructure(data.ProteinDataset):
    """
    Secondary structure labels for a set of proteins determined by the local structures 
    of protein residues in their natural state

    Statistics:
        - #Train: 8,678
        - #Valid: 2,170
        - #Test: 513

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/secondary_structure.tar.gz"
    md5 = "2f61e8e09c215c032ef5bc8b910c8e97"
    splits = ["train", "valid", "casp12", "ts115", "cb513"]
    target_fields = ["ss3", "valid_mask"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        lmdb_files = [os.path.join(data_path, "secondary_structure/secondary_structure_%s.lmdb" % split) 
                                    for split in self.splits]

        self.load_lmdbs(lmdb_files, target_fields=self.target_fields, verbose=verbose, **kwargs)

    def get_item(self, index):
        if self.lazy:
            graph = data.Protein.from_sequence(self.sequences[index], **self.kwargs)
        else:
            graph = self.data[index]
        with graph.residue():
            target = torch.as_tensor(self.targets["ss3"][index], dtype=torch.long)
            graph.target = target
            mask = torch.as_tensor(self.targets["valid_mask"][index], dtype=torch.bool)
            graph.mask = mask
        item = {"graph": graph}
        if self.transform:
            item = self.transform(item)
        return item

    def split(self, keys=None):
        keys = keys or self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits