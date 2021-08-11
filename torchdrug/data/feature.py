import warnings

from rdkit import Chem
from rdkit.Chem import AllChem

from torchdrug.core import Registry as R


atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
degree_vocab = range(7)
num_hs_vocab = range(7)
formal_charge_vocab = range(-5, 6)
chiral_tag_vocab = range(4)
total_valence_vocab = range(8)
num_radical_vocab = range(8)
hybridization_vocab = range(len(Chem.rdchem.HybridizationType.values))

bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
bond_stereo_vocab = range(len(Chem.rdchem.BondStereo.values))


def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1

    return feature


# TODO: this one is too slow
@R.register("features.atom.default")
def atom_default(atom):
    """Default atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        
        GetChiralTag(): one-hot embedding for atomic chiral tag 
        
        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs
        
        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule
        
        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetNumRadicalElectrons(): one-hot embedding for the number of radical electrons on the atom
        
        GetHybridization(): one-hot embedding for the atom's hybridization
        
        GetIsAromatic(): whether the atom is aromatic
        
        IsInRing(): whether the atom is in a ring
        
        atom_position(): the 3D position of the atom
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetChiralTag(), chiral_tag_vocab) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab) + \
           onehot(atom.GetNumRadicalElectrons(), num_radical_vocab) + \
           onehot(atom.GetHybridization(), hybridization_vocab) + \
           [atom.GetIsAromatic(), atom.IsInRing()] + \
           atom_position(atom)


@R.register("features.atom.center_identification")
def atom_center_identification(atom):
    """Reaction center identification atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        
        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs
        
        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom
        
        GetIsAromatic(): whether the atom is aromatic
        
        IsInRing(): whether the atom is in a ring
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab) + \
           [atom.GetIsAromatic(), atom.IsInRing()]


@R.register("features.atom.synthon_completion")
def atom_synthon_completion(atom):
    """Synthon completion atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs
        
        IsInRing(): whether the atom is in a ring
        
        IsInRingSize(3, 4, 5, 6): whether the atom is in a ring of a particular size
        
        IsInRing() and not IsInRingSize(3, 4, 5, 6): whether the atom is in a ring and not in a ring of 3, 4, 5, 6
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           [atom.IsInRing(), atom.IsInRingSize(3), atom.IsInRingSize(4),
            atom.IsInRingSize(5), atom.IsInRingSize(6), 
            atom.IsInRing() and (not atom.IsInRingSize(3)) and (not atom.IsInRingSize(4)) \
            and (not atom.IsInRingSize(5)) and (not atom.IsInRingSize(6))]


@R.register("features.atom.symbol")
def atom_symbol(atom):
    """Symbol atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)


@R.register("features.atom.explicit_property_prediction")
def atom_explicit_property_prediction(atom):
    """Explicit property prediction atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetDegree(): one-hot embedding for the degree of the atom in the molecule

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom
        
        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule
        
        GetIsAromatic(): whether the atom is aromatic
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab) + \
           [atom.GetIsAromatic()]


@R.register("features.atom.property_prediction")
def atom_property_prediction(atom):
    """Property prediction atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        
        GetDegree(): one-hot embedding for the degree of the atom in the molecule
        
        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom
        
        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule
        
        GetIsAromatic(): whether the atom is aromatic
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab, allow_unknown=True) + \
           [atom.GetIsAromatic()]


@R.register("features.atom.position")
def atom_position(atom):
    """
    Atom position.
    Return 3D position if available, otherwise 2D position is returned.
    """
    mol = atom.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    pos = conformer.GetAtomPosition(atom.GetIdx())
    return [pos.x, pos.y, pos.z]


@R.register("features.atom.pretrain")
def atom_pretrain(atom):
    """Atom feature for pretraining.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        
        GetChiralTag(): one-hot embedding for atomic chiral tag
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetChiralTag(), chiral_tag_vocab)


@R.register("features.bond.default")
def bond_default(bond):
    """Default bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond
        
        GetBondDir(): one-hot embedding for the direction of the bond
        
        GetStereo(): one-hot embedding for the stereo configuration of the bond
        
        GetIsConjugated(): whether the bond is considered to be conjugated
        
        bond_length: the length of the bond
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + \
           onehot(bond.GetBondDir(), bond_dir_vocab) + \
           onehot(bond.GetStereo(), bond_stereo_vocab) + \
           [int(bond.GetIsConjugated())] + \
           bond_length(bond)


@R.register("features.bond.length")
def bond_length(bond):
    """Bond length"""
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    h = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
    t = conformer.GetAtomPosition(bond.GetEndAtomIdx())
    return [h.Distance(t)]


@R.register("features.bond.property_prediction")
def bond_property_prediction(bond):
    """Property prediction bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond
        
        GetIsConjugated(): whether the bond is considered to be conjugated
        
        IsInRing(): whether the bond is in a ring
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + \
           [int(bond.GetIsConjugated()), bond.IsInRing()]


@R.register("features.bond.pretrain")
def bond_pretrain(bond):
    """Bond feature for pretraining.

    Features:
        GetBondType(): one-hot embedding for the type of the bond
        
        GetBondDir(): one-hot embedding for the direction of the bond
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + \
           onehot(bond.GetBondDir(), bond_dir_vocab)


@R.register("features.molecule.ecfp")
def ExtendedConnectivityFingerprint(mol, radius=2, length=1024):
    """Extended Connectivity Fingerprint molecule feature.

    Features:
        GetMorganFingerprintAsBitVect(): a Morgan fingerprint for a molecule as a bit vector
    """
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
    return list(ecfp)


@R.register("features.molecule.default")
def molecule_default(mol):
    """Default molecule feature."""
    return ExtendedConnectivityFingerprint(mol)

ECFP = ExtendedConnectivityFingerprint


__all__ = [
    "atom_default", "atom_center_identification", "atom_synthon_completion",
    "atom_symbol", "atom_explicit_property_prediction", "atom_property_prediction",
    "atom_position", "atom_pretrain",
    "bond_default", "bond_length", "bond_property_prediction", "bond_pretrain",
    "ExtendedConnectivityFingerprint", "molecule_default",
    "ECFP",
]