import os

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.WN18")
class WN18(data.KnowledgeGraphDataset):
    """
    WordNet knowledge base.

    Statistics:
        - #Entity: 40,943
        - #Relation: 18
        - #Triplet: 151,442

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    urls = [
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18/train.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18/valid.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18/test.txt",
    ]
    md5s = [
        "7d68324d293837ac165c3441a6c8b0eb",
        "f4f66fec0ca83b5ebe7ad7003404e61d",
        "b035247a8916c7ec3443fa949e1ff02c"
    ]

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url, md5 in zip(self.urls, self.md5s):
            save_file = "wn18_%s" % os.path.basename(url)
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


@R.register("datasets.WN18RR")
class WN18RR(data.KnowledgeGraphDataset):
    """
    A filtered version of WN18 dataset without trivial cases.

    Statistics:
        - #Entity: 40,943
        - #Relation: 11
        - #Triplet: 93,003

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    urls = [
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18rr/train.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18rr/valid.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/wn18rr/test.txt",
    ]
    md5s = [
        "35e81af3ae233327c52a87f23b30ad3c",
        "74a2ee9eca9a8d31f1a7d4d95b5e0887",
        "2b45ba1ba436b9d4ff27f1d3511224c9"
    ]

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url, md5 in zip(self.urls, self.md5s):
            save_file = "wn18rr_%s" % os.path.basename(url)
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