import os

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.YAGO310")
class YAGO310(data.KnowledgeGraphDataset):
    """
    Subset of YAGO3 knowledge base for knowledge graph reasoning.

    Statistics:
        - #Entity: 123,182
        - #Relation: 37
        - #Triplet: 1,089,040

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    urls = [
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/train.txt",
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/valid.txt",
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/test.txt",
    ]
    md5s = [
        "a9da8f583ec3920570eeccf07199229a",
        "2d679a906f2b1ac29d74d5c948c1ad09",
        "14bf97890b2fee774dbce5f326acd189"
    ]

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        txt_files = []
        for url, md5 in zip(self.urls, self.md5s):
            save_file = "yago310_%s" % os.path.basename(url)
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
