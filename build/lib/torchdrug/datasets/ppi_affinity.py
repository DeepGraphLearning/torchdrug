import os

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.PPIAffinity")
@utils.copy_args(data.ProteinPairDataset.load_lmdbs, ignore=("sequence_field", "target_fields"))
class PPIAffinity(data.ProteinPairDataset):
    r"""
    The binding affinity values measured by :math:`p_{K_d}` between two proteins.

    Statistics:
        - #Train: 2,127
        - #Valid: 212
        - #Test: 343

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/ppidata/ppi_affinity.zip"
    md5 = "d114907fd20c75820e41881f8901e9e4"
    splits = ["train", "valid", "test"]
    target_fields = ["interaction"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        lmdb_files = [os.path.join(data_path, "ppi_affinity/ppi_affinity_%s.lmdb" % split)
                      for split in self.splits]

        self.load_lmdbs(lmdb_files, sequence_field=["primary_1", "primary_2"], target_fields=self.target_fields,
                        verbose=verbose, **kwargs)

    def split(self, keys=None):
        keys = keys or self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits

    def get_item(self, index):
        if self.lazy:
            graph1 = data.Protein.from_sequence(self.sequences[index][0], **self.kwargs)
            graph2 = data.Protein.from_sequence(self.sequences[index][1], **self.kwargs)
        else:
            graph1 = self.data[index][0]
            graph2 = self.data[index][1]
        item = {"graph1": graph1, "graph2": graph2}
        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item