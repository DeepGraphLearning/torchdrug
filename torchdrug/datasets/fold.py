import os

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.Fold")
@utils.copy_args(data.ProteinDataset.load_lmdbs, ignore=("target_fields",))
class Fold(data.ProteinDataset):
    """
    Fold labels for a set of proteins determined by the global structural topology.

    Statistics:
        - #Train: 12,312
        - #Valid: 736
        - #Test: 718

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/remote_homology.tar.gz"
    md5 = "1d687bdeb9e3866f77504d6079eed00a"
    splits = ["train", "valid", "test_fold_holdout", "test_family_holdout", "test_superfamily_holdout"]
    target_fields = ["fold_label"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        lmdb_files = [os.path.join(data_path, "remote_homology/remote_homology_%s.lmdb" % split)
                      for split in self.splits]

        self.load_lmdbs(lmdb_files, target_fields=self.target_fields, verbose=verbose, **kwargs)

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