import os

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.Fluorescence")
@utils.copy_args(data.ProteinDataset.load_lmdbs, ignore=("target_fields",))
class Fluorescence(data.ProteinDataset):
    """
    The fitness values of a set of green fluorescent protein mutants.

    Statistics:
        - #Train: 21,446
        - #Valid: 5,362
        - #Test: 27,217

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz"
    md5 = "d63d1d51ec8c20ff0d981e4cbd67457a"
    splits = ["train", "valid", "test"]
    target_fields = ["log_fluorescence"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        lmdb_files = [os.path.join(data_path, "fluorescence/fluorescence_%s.lmdb" % split)
                      for split in self.splits]

        self.load_lmdbs(lmdb_files, target_fields=self.target_fields, verbose=verbose, **kwargs)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits