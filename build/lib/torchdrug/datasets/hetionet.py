import os

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.Hetionet")
class Hetionet(data.KnowledgeGraphDataset):
    """
    Hetionet for knowledge graph reasoning.

    Statistics:
        - #Entity: 45,158
        - #Relation: 24
        - #Triplet: 2,025,177

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    urls = [
        "https://www.dropbox.com/s/y47bt9oq57h6l5k/train.txt?dl=1",
        "https://www.dropbox.com/s/a0pbrx9tz3dgsff/valid.txt?dl=1",
        "https://www.dropbox.com/s/4dhrvg3fyq5tnu4/test.txt?dl=1",
    ]
    md5s = [
        "6e58915d70ce6d9389c6e4785245e0b3",
        "77f15fac4f8170b836392a5b1d315afa",
        "e8877aafe89d0c9b9c1efb9027cb7226"
    ]

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url, md5 in zip(self.urls, self.md5s):
            save_file = "hetionet_%s.txt" % os.path.splitext(os.path.basename(url))[0]
            txt_file = utils.download(url, self.path, save_file=save_file, md5=md5)
            txt_files.append(txt_file)

        self.load_tsvs(txt_files, verbose=verbose)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

