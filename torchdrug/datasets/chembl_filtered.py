import os
import random
import pickle
from tqdm import tqdm
import numpy as np
from itertools import chain
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from torchdrug import data, datasets, utils
from torchdrug.core import Registry as R
from torchdrug.utils import doc


@R.register("datasets.ChEMBLFiltered")
@doc.copy_args(data.MoleculeDataset.load_csv, ignore=("smiles_field", "target_fields"))
class ChEMBLFiltered(data.MoleculeDataset):
    """
    Statistics:
        - #Molecule: 430,711
        - #Regression task: 1,310

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip"
    md5 = "e95da4dffa0fdb1d4af2726bdf8c23e0"

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        zip_file_name = utils.download(self.url, path, md5=self.md5)
        utils.extract(zip_file_name)

        file_name = preprocess('./temp/chem_dataset')
        print('file_name is {}'.format(file_name))

        self.target_fields = ["target_{}".format(i) for i in range(1310)]

        self.load_csv(file_name, smiles_field="smiles", target_fields=self.target_fields,
                      verbose=verbose, **kwargs)
        return


def split_rdkit_mol_obj(mol):
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def get_largest_mol(mol_list):
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def create_standardized_mol_id(smiles):
    """
    :param smiles:
    :return: inchi
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol != None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return


def _load_chembl_with_labels_dataset(root_path):
    # 1. load folds and labels
    f=open(os.path.join(root_path, 'folds0.pckl'), 'rb')
    folds=pickle.load(f)
    f.close()

    f=open(os.path.join(root_path, 'labelsHard.pckl'), 'rb')
    targetMat=pickle.load(f)
    sampleAnnInd=pickle.load(f)
    targetAnnInd=pickle.load(f)
    f.close()

    targetMat=targetMat
    targetMat=targetMat.copy().tocsr()
    targetMat.sort_indices()
    targetAnnInd=targetAnnInd
    targetAnnInd=targetAnnInd-targetAnnInd.min()

    folds=[np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed=targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # # num positive examples in each of the 1310 targets
    trainPosOverall=np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
    # # num negative examples in each of the 1310 targets
    trainNegOverall=np.array([np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])
    # dense array containing the labels for the 456331 molecules and 1310 targets
    denseOutputData=targetMat.A # possible values are {-1, 0, 1}

    # 2. load structures
    f=open(os.path.join(root_path, 'chembl20LSTM.pckl'), 'rb')
    rdkitArr=pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    print('preprocessing')

    for i, m in enumerate(tqdm(rdkitArr)):
        if m == None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in preprocessed_rdkitArr]   # bc some empty mol in the

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    return smiles_list, preprocessed_rdkitArr, folds, denseOutputData


def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def DataSplit(dataset):
    frac_train, frac_valid, frac_test = 0.8, 0.1, 0.1

    scaffold_list = []
    all_scaffolds = defaultdict(list)
    for idx, smiles in enumerate(dataset.smiles_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=True)
        scaffold_list.append(scaffold)
        all_scaffolds[scaffold].append(idx)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(scaffold_list)
    valid_cutoff = (frac_train + frac_valid) * len(scaffold_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    return train_idx, valid_idx, test_idx


def preprocess(dir_path):
    config_dataset2dataset = {
        'tox21': datasets.Tox21,
        'toxcast': datasets.ToxCast,
        'clintox': datasets.ClinTox,
        'bbbp': datasets.BBBP,
        'bace': datasets.BACE,
        'sider': datasets.SIDER,
        'muv': datasets.MUV,
        'hiv': datasets.HIV,
        'delaney': datasets.Delaney,
        'lipophilicity': datasets.Lipophilicity,
        'freesolv': datasets.FreeSolv,
    }

    random.seed(0)
    path = '{}/dataset'.format(dir_path)
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    output_file = '{}/chembl_filtered.csv'.format(dir_path)
    if os.path.exists(output_file):
        return output_file

    downstream_task_list = [
        'bace', 'bbbp', 'clintox', 'delaney', 'freesolv', 'hiv', 'lipophilicity', 'muv', 'sider', 'tox21', 'toxcast']

    downstream_inchi_set = set()
    for downstream_task in downstream_task_list:
        dataset = config_dataset2dataset[downstream_task]('./datasets', verbose=True)
        _, valid_indices, test_indices = DataSplit(dataset=dataset)

        indices = valid_indices + test_indices
        smiles_list = np.array(dataset.smiles_list)
        downstream_smiles = smiles_list[indices]

        print(downstream_task, '\t', len(downstream_smiles))

        remove_smiles = downstream_smiles

        downstream_inchis = []
        for smiles in remove_smiles:
            species_list = smiles.split('.')
            for s in species_list:
                inchi = create_standardized_mol_id(s)
                downstream_inchis.append(inchi)
        downstream_inchi_set.update(downstream_inchis)

    smiles_list, rdkit_mol_objs, folds, labels = _load_chembl_with_labels_dataset('{}/chembl_filtered/raw'.format(path))

    print('processing')
    filtered_smiles_list, filtered_label_list = [], []
    for idx, rdkit_mol in enumerate(tqdm(rdkit_mol_objs)):
        if rdkit_mol != None:
            mw = Descriptors.MolWt(rdkit_mol)
            if 50 <= mw <= 900:
                inchi = create_standardized_mol_id(smiles_list[idx])
                if inchi != None and inchi not in downstream_inchi_set:
                    filtered_smiles_list.append(smiles_list[idx])
                    filtered_label_list.append(labels[idx])
    print('filtered_smiles_list\t', len(filtered_smiles_list))

    min_, max_, valid_count = 1e10, -1e10, 0
    for idx, smiles in enumerate(tqdm(filtered_smiles_list)):
        mol = Chem.MolFromSmiles(smiles)
        num = mol.GetNumAtoms()
        if num == 0:
            print('invalid? {}'.format(idx))
        else:
            valid_count += 1
            min_ = min(min_, num)
            max_ = max(max_, num)

    target_fields = ['target_{}'.format(i) for i in range(labels.shape[1])]
    head = ['smiles'] + target_fields

    print('saving to {}'.format(output_file))

    with open(output_file, 'w') as f:
        print(','.join(head), file=f)
        for smiles, label_list in zip(filtered_smiles_list, filtered_label_list):
            line = smiles
            for x in label_list:
                if x == 1:
                    line = '{},1'.format(line)
                elif x == -1:
                    line = '{},0'.format(line)
                elif x == 0:
                    line = '{},'.format(line)
                else:
                    raise ValueError('label {} is invalid'.format(x))
            print(line, file=f)
    return output_file
