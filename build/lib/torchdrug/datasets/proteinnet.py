import os

import torch
from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.ProteinNet")
@utils.copy_args(data.ProteinDataset.load_lmdbs, ignore=("target_fields",))
class ProteinNet(data.ProteinDataset):
    """
    A set of proteins with 3D structures for the contact prediction task.

    Statistics:
        - #Train: 25,299
        - #Valid: 224
        - #Test: 40

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/data/proteinnet.tar.gz"
    md5 = "ab44ab201b1570c0171a2bba9eb4d389"
    splits = ["train", "valid", "test"]
    target_fields = ["tertiary", "valid_mask"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        lmdb_files = [os.path.join(data_path, "proteinnet/proteinnet_%s.lmdb" % split)
                      for split in self.splits]

        self.load_lmdbs(lmdb_files, target_fields=self.target_fields, verbose=verbose, **kwargs)

    def get_item(self, index):
        if self.lazy:
            graph = data.Protein.from_sequence(self.sequences[index], **self.kwargs)
        else:
            graph = self.data[index]
        with graph.residue():
            residue_position = torch.as_tensor(self.targets["tertiary"][index], dtype=torch.float)
            graph.residue_position = residue_position
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