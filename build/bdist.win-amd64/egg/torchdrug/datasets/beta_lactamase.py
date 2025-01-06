import os

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.BetaLactamase")
@utils.copy_args(data.ProteinDataset.load_lmdbs, ignore=("target_fields",))
class BetaLactamase(data.ProteinDataset):
    """
    The activity values of first-order mutants of the TEM-1 beta-lactamase protein.

    Statistics:
        - #Train: 4,158
        - #Valid: 520
        - #Test: 520

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/beta_lactamase.tar.gz"
    md5 = "65766a3969cc0e94b101d4063d204ba4"
    splits = ["train", "valid", "test"]
    target_fields = ["scaled_effect1"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        lmdb_files = [os.path.join(data_path, "beta_lactamase/beta_lactamase_%s.lmdb" % split)
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