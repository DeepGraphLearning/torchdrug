import os

from rdkit import Chem

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.PDBBind")
@utils.copy_args(data.ProteinLigandDataset.load_lmdbs, ignore=("sequence_field", "smiles_field", "target_fields"))
class PDBBind(data.ProteinLigandDataset):
    """
    The PDBbind-2019 dataset with binding affinity indicating the interaction strength 
    between pairs of protein and ligand.

    Statistics:
        - #Train: 16,436
        - #Valid: 937
        - #Test: 285

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/pdbind.tar.gz"
    md5 = "5f5b3d2cd5f5a5fcf9e6da922850f4a0"
    splits = ["train", "valid", "test"]
    target_fields = ["affinity"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        lmdb_files = [os.path.join(data_path, "pdbind/pdbind_%s.lmdb" % split)
                      for split in self.splits]

        self.load_lmdbs(lmdb_files, sequence_field="target", smiles_field="drug",
                        target_fields=self.target_fields, verbose=verbose, **kwargs)

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
            graph1 = data.Protein.from_sequence(self.sequences[index], **self.kwargs)
            mol = Chem.MolFromSmiles(self.smiles[index])
            if not mol:
                graph2 = None
            else:
                graph2 = data.Molecule.from_molecule(mol, **self.kwargs)
        else:
            graph1 = self.data[index][0]
            graph2 = self.data[index][1]
        item = {"graph1": graph1, "graph2": graph2}
        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item