import os
import re
import csv

from tqdm import tqdm

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.PubMed")
class PubMed(data.NodeClassificationDataset):
    """
    A citation network of scientific publications with TF-IDF word features.

    Statistics:
        - #Node: 19,717
        - #Edge: 44,338
        - #Class: 3

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    url = "https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz"
    md5 = "9fa24b917990c47e264a94079b9599fe"

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        node_file = utils.extract(zip_file, "Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab")
        edge_file = utils.extract(zip_file, "Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab")

        inv_node_vocab = {}
        node_feature = []
        node_label = []

        with open(node_file, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" % node_file, utils.get_line_count(node_file)))
            _ = next(reader)
            fields = next(reader)
            group, = re.match(r"cat=(\S+):label", fields[0]).groups()
            label_tokens = group.split(",")
            inv_label_vocab = {token: i for i, token in enumerate(label_tokens)}
            inv_feature_vocab = {}
            for field in fields[1:]:
                match = re.match(r"numeric:(\S+):0\.0", field)
                if not match:
                    continue
                feature_token, = match.groups()
                inv_feature_vocab[feature_token] = len(inv_feature_vocab)

            for tokens in reader:
                node_token = tokens[0]
                label_token, = re.match(r"label=(\S+)", tokens[1]).groups()
                feature = [0] * len(inv_feature_vocab)
                inv_node_vocab[node_token] = len(inv_node_vocab)
                for token in tokens[2:]:
                    match = re.match(r"(\S+)=([0-9.]+)", token)
                    if not match:
                        continue
                    feature_token, value = match.groups()
                    feature[inv_feature_vocab[feature_token]] = utils.literal_eval(value)
                label = inv_label_vocab[label_token]
                node_feature.append(feature)
                node_label.append(label)

        edge_list = []

        with open(edge_file, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" % edge_file, utils.get_line_count(edge_file)))
            _ = next(reader)
            _ = next(reader)
            for tokens in reader:
                h_token, = re.match(r"paper:(\S+)", tokens[1]).groups()
                t_token, = re.match(r"paper:(\S+)", tokens[3]).groups()
                if h_token not in inv_node_vocab:
                    inv_node_vocab[h_token] = len(inv_node_vocab)
                h = inv_node_vocab[h_token]
                if t_token not in inv_node_vocab:
                    inv_node_vocab[t_token] = len(inv_node_vocab)
                t = inv_node_vocab[t_token]
                edge_list.append((h, t))

        self.load_edge(edge_list, node_feature, node_label, inv_node_vocab=inv_node_vocab,
                       inv_label_vocab=inv_label_vocab)