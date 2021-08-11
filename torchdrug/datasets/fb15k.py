import os

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.FB15k")
class FB15k(data.KnowledgeGraphDataset):
    """
    Subset of Freebase knowledge base for knowledge graph reasoning.

    Statistics:
        - #Entity: 14,951
        - #Relation: 1,345
        - #Triplet: 592,213

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    urls = [
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k/train.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k/valid.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k/test.txt",
    ]
    md5s = [
        "5a87195e68d7797af00e137a7f6929f2",
        "275835062bb86a86477a3c402d20b814",
        "71098693b0efcfb8ac6cd61cf3a3b505"
    ]

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url, md5 in zip(self.urls, self.md5s):
            save_file = "fb15k_%s" % os.path.basename(url)
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


@R.register("datasets.FB15k237")
class FB15k237(data.KnowledgeGraphDataset):
    """
    A filtered version of FB15k dataset without trivial cases.

    Statistics:
        - #Entity: 14,541
        - #Relation: 237
        - #Triplet: 310,116

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    urls = [
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k-237/train.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k-237/valid.txt",
        "https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k-237/test.txt",
    ]
    md5s = [
        "c05b87b9ac00f41901e016a2092d7837",
        "6a94efd530e5f43fcf84f50bc6d37b69",
        "f5bdf63db39f455dec0ed259bb6f8628"
    ]

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url, md5 in zip(self.urls, self.md5s):
            save_file = "fb15k237_%s" % os.path.basename(url)
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