"""
GuacaMol Benchmark Dataset 
Author: Aditya Vartak
"""

import os
from collections import defaultdict
from torch.utils import data as torch_data
from torchdrug import data, utils
from torchdrug.core import Registry as R
from torchdrug.utils import doc
import shlex
import subprocess
import csv


@R.register("datasets.GuacaMol")
@doc.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class GuacaMol(data.MoleculeDataset):
    """
    Subset of ChemBL database for molecule generation.
    Benchmark Dataset for De novo Molecular design 
    This dataset doesn't contain any label information.

    Statistics:
        #Molecule: 1591380
        #task: 1
    Parameters:
        path (str): path for the CSV dataset
        verbose (int, optional): output verbose level
        **kwargs

    """
    target_fields = ["source"]
    def __init__(self,path=None,verbose=False,**kwargs):
        process = subprocess.Popen(shlex.split("python -m guacamol.data.get_data -o ."),
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stderr)
        print("Downloaded files")
       
        
        smiles_gz = "chembl_24_1_chemreps.txt.gz"
        train_smiles_path = 'chembl24_canon_train.smiles'
        valid_smiles_path = 'chembl24_canon_dev-valid.smiles'
        test_smiles_path = 'chembl24_canon_test.smiles'
        path = 'output.csv'
        path = self.smiles_to_csv(train_smiles_path,valid_smiles_path,test_smiles_path,path)

        self.load_csv(path, smiles_field="smiles", target_fields=self.target_fields,
                      lazy=True, verbose=verbose, **kwargs)

        process = subprocess.Popen(shlex.split(f"rm {smiles_gz} {train_smiles_path} {valid_smiles_path} {test_smiles_path} {path}"),
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)



    def smiles_to_csv(self,train_smiles_path,valid_smiles_path,test_smiles_path,path_to_save):
        final_data = []
        print(train_smiles_path)
        with open(train_smiles_path,'r') as f:
            train_smiles = f.readlines()
            final_data.extend([[i,'valid'] for i in train_smiles])
        with open(valid_smiles_path,'r') as f:
            valid_smiles = f.readlines()
            final_data.extend([[i,'valid'] for i in valid_smiles])

        with open(test_smiles_path,'r') as f:
            test_smiles = f.readlines()
            final_data.extend([[i,'valid'] for i in test_smiles])

        with open(path_to_save, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles","source"])
            writer.writerows(final_data)
        
        return path_to_save


                


