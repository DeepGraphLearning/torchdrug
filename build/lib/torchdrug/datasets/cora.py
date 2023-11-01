import os

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.Cora")
class Cora(data.NodeClassificationDataset):
    """
    A citation network of scientific publications with binary word features.

    Statistics:
        - #Node: 2,708
        - #Edge: 5,429
        - #Class: 7

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    md5 = "2fc040bee8ce3d920e4204effd1e9214"

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        node_file = utils.extract(zip_file, "cora/cora.content")
        edge_file = utils.extract(zip_file, "cora/cora.cites")

        self.load_tsv(node_file, edge_file, verbose=verbose)