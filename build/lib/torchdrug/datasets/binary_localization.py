import os

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.BinaryLocalization")
@utils.copy_args(data.ProteinDataset.load_lmdbs, ignore=("target_fields",))
class BinaryLocalization(data.ProteinDataset):
    """
    Simpler version of the Subcellular Localization with binary labels indicating
    whether a protein is membrane-bound or soluble.

    Statistics:
        - #Train: 5,161
        - #Valid: 1,727
        - #Test: 1,746

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization_2.tar.gz"
    md5 = "5d2309bf1c0c2aed450102578e434f4e"
    splits = ["train", "valid", "test"]
    target_fields = ["localization"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        lmdb_files = [os.path.join(data_path, "subcellular_localization_2/subcellular_localization_2_%s.lmdb" % split)
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