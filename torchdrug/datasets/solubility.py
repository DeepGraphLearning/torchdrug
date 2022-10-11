import os

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.Solubility")
@utils.copy_args(data.ProteinDataset.load_lmdbs, ignore=("target_fields",))
class Solubility(data.ProteinDataset):
    """
    Proteins with binary labels indicating their solubility.

    Statistics:
        - #Train: 62,478
        - #Valid: 6,942
        - #Test: 1,999

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/solubility.tar.gz"
    md5 = "8a8612b7bfa2ed80375db6e465ccf77e"
    splits = ["train", "valid", "test"]
    target_fields = ["solubility"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        lmdb_files = [os.path.join(data_path, "solubility/solubility_%s.lmdb" % split)
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