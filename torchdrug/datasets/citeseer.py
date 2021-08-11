import os

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.CiteSeer")
class CiteSeer(data.NodeClassificationDataset):
    """
    A citation network of scientific publications with binary word features.

    Statistics:
        - #Node: 3,327
        - #Edge: 8,059
        - #Class: 6

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    url = "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz"
    md5 = "c8ded8ed395b31899576bfd1e91e4d6e"

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        node_file = utils.extract(zip_file, "citeseer/citeseer.content")
        edge_file = utils.extract(zip_file, "citeseer/citeseer.cites")

        self.load_tsv(node_file, edge_file, verbose=verbose)