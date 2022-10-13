import math
import warnings
from copy import copy
from collections import Sequence

from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch_scatter import scatter_add, scatter_min

from torchdrug import utils
from torchdrug.data import constant, Graph, PackedGraph
from torchdrug.core import Registry as R
from torchdrug.data.rdkit import draw
from torchdrug.utils import pretty

plt.switch_backend("agg")


class Molecule(Graph):
    """
    Molecules with predefined chemical features.

    By nature, molecules are undirected graphs. Each bond is stored as two directed edges in this class.

    .. warning::

        This class doesn't enforce any order on edges.

    Parameters:
        edge_list (array_like, optional): list of edges of shape :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out, bond_type).
        atom_type (array_like, optional): atom types of shape :math:`(|V|,)`
        bond_type (array_like, optional): bond types of shape :math:`(|E|,)`
        formal_charge (array_like, optional): formal charges of shape :math:`(|V|,)`
        explicit_hs (array_like, optional): number of explicit hydrogens of shape :math:`(|V|,)`
        chiral_tag (array_like, optional): chirality tags of shape :math:`(|V|,)`
        radical_electrons (array_like, optional): number of radical electrons of shape :math:`(|V|,)`
        atom_map (array_likeb optional): atom mappings of shape :math:`(|V|,)`
        bond_stereo (array_like, optional): bond stereochem of shape :math:`(|E|,)`
        stereo_atoms (array_like, optional): ids of stereo atoms of shape :math:`(|E|,)`
    """

    bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
    atom2valence = {1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 5, 16: 6, 17: 1, 35: 1, 53: 7}
    bond2valence = [1, 2, 3, 1.5]
    id2bond = {v: k for k, v in bond2id.items()}
    empty_mol = Chem.MolFromSmiles("")
    dummy_mol = Chem.MolFromSmiles("CC")

    def __init__(self, edge_list=None, atom_type=None, bond_type=None, atom_feature=None, bond_feature=None,
                 mol_feature=None, formal_charge=None, explicit_hs=None, chiral_tag=None, radical_electrons=None,
                 atom_map=None, bond_stereo=None, stereo_atoms=None, node_position=None, **kwargs):
        if "num_relation" not in kwargs:
            kwargs["num_relation"] = len(self.bond2id)
        super(Molecule, self).__init__(edge_list=edge_list, **kwargs)
        atom_type, bond_type = self._standarize_atom_bond(atom_type, bond_type)

        formal_charge = self._standarize_attribute(formal_charge, self.num_node)
        explicit_hs = self._standarize_attribute(explicit_hs, self.num_node)
        chiral_tag = self._standarize_attribute(chiral_tag, self.num_node)
        radical_electrons = self._standarize_attribute(radical_electrons, self.num_node)
        atom_map = self._standarize_attribute(atom_map, self.num_node)
        bond_stereo = self._standarize_attribute(bond_stereo, self.num_edge)
        stereo_atoms = self._standarize_attribute(stereo_atoms, (self.num_edge, 2))
        if node_position is not None:
            node_position = torch.as_tensor(node_position, dtype=torch.float, device=self.device)

        with self.atom():
            if atom_feature is not None:
                self.atom_feature = torch.as_tensor(atom_feature, device=self.device)
            self.atom_type = atom_type
            self.formal_charge = formal_charge
            self.explicit_hs = explicit_hs
            self.chiral_tag = chiral_tag
            self.radical_electrons = radical_electrons
            self.atom_map = atom_map
            if node_position is not None:
                self.node_position = node_position

        with self.bond():
            if bond_feature is not None:
                self.bond_feature = torch.as_tensor(bond_feature, device=self.device)
            self.bond_type = bond_type
            self.bond_stereo = bond_stereo
            self.stereo_atoms = stereo_atoms

        with self.mol():
            if mol_feature is not None:
                self.mol_feature = torch.as_tensor(mol_feature, device=self.device)

    def _standarize_atom_bond(self, atom_type, bond_type):
        if atom_type is None:
            raise ValueError("`atom_type` should be provided")
        if bond_type is None:
            raise ValueError("`bond_type` should be provided")

        atom_type = torch.as_tensor(atom_type, dtype=torch.long, device=self.device)
        bond_type = torch.as_tensor(bond_type, dtype=torch.long, device=self.device)
        return atom_type, bond_type

    def _standarize_attribute(self, attribute, size, dtype=torch.long, default=0):
        if attribute is not None:
            attribute = torch.as_tensor(attribute, dtype=dtype, device=self.device)
        else:
            if isinstance(size, torch.Tensor):
                size = size.tolist()
            if not isinstance(size, Sequence):
                size = [size]
            attribute = torch.full(size, default, dtype=dtype, device=self.device)
        return attribute

    @classmethod
    def _standarize_option(cls, option):
        if option is None:
            option = []
        elif isinstance(option, str):
            option = [option]
        return option

    def _check_no_stereo(self):
        if (self.bond_stereo > 0).any():
            warnings.warn("Try to apply masks on molecules with stereo bonds. This may produce invalid molecules. "
                          "To discard stereo information, call `mol.bond_stereo[:] = 0` before applying masks.")

    def _maybe_num_node(self, edge_list):
        if len(edge_list):
            return edge_list[:, :2].max().item() + 1
        else:
            return 0

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_molecule(cls, mol, atom_feature="default", bond_feature="default", mol_feature=None,
                      with_hydrogen=False, kekulize=False):
        """
        Create a molecule from an RDKit object.

        Parameters:
            mol (rdchem.Mol): molecule
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            with_hydrogen (bool, optional): store hydrogens in the molecule graph.
                By default, hydrogens are dropped
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if mol is None:
            mol = cls.empty_mol
        # some RDKit operations are in-place
        # copy the object to avoid undesired behavior in the caller
        mol = copy(mol)
        if with_hydrogen:
            mol = Chem.AddHs(mol)
        if kekulize:
            Chem.Kekulize(mol)

        atom_feature = cls._standarize_option(atom_feature)
        bond_feature = cls._standarize_option(bond_feature)
        mol_feature = cls._standarize_option(mol_feature)

        atom_type = []
        formal_charge = []
        explicit_hs = []
        chiral_tag = []
        radical_electrons = []
        atom_map = []
        _atom_feature = []
        dummy_atom = copy(cls.dummy_mol).GetAtomWithIdx(0)
        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())] + [dummy_atom]
        if mol.GetNumConformers() > 0:
            node_position = torch.tensor(mol.GetConformer().GetPositions())
        else:
            node_position = None
        for atom in atoms:
            atom_type.append(atom.GetAtomicNum())
            formal_charge.append(atom.GetFormalCharge())
            explicit_hs.append(atom.GetNumExplicitHs())
            chiral_tag.append(atom.GetChiralTag())
            radical_electrons.append(atom.GetNumRadicalElectrons())
            atom_map.append(atom.GetAtomMapNum())
            feature = []
            for name in atom_feature:
                func = R.get("features.atom.%s" % name)
                feature += func(atom)
            _atom_feature.append(feature)
        atom_type = torch.tensor(atom_type)[:-1]
        atom_map = torch.tensor(atom_map)[:-1]
        formal_charge = torch.tensor(formal_charge)[:-1]
        explicit_hs = torch.tensor(explicit_hs)[:-1]
        chiral_tag = torch.tensor(chiral_tag)[:-1]
        radical_electrons = torch.tensor(radical_electrons)[:-1]
        if len(atom_feature) > 0:
            _atom_feature = torch.tensor(_atom_feature)[:-1]
        else:
            _atom_feature = None

        edge_list = []
        bond_type = []
        bond_stereo = []
        stereo_atoms = []
        _bond_feature = []
        dummy_bond = copy(cls.dummy_mol).GetBondWithIdx(0)
        bonds = [mol.GetBondWithIdx(i) for i in range(mol.GetNumBonds())] + [dummy_bond]
        for bond in bonds:
            type = str(bond.GetBondType())
            stereo = bond.GetStereo()
            if stereo:
                _atoms = [a for a in bond.GetStereoAtoms()]
            else:
                _atoms = [0, 0]
            if type not in cls.bond2id:
                continue
            type = cls.bond2id[type]
            h, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list += [[h, t, type], [t, h, type]]
            # always explicitly store aromatic bonds, no matter kekulize or not
            if bond.GetIsAromatic():
                type = cls.bond2id["AROMATIC"]
            bond_type += [type, type]
            bond_stereo += [stereo, stereo]
            stereo_atoms += [_atoms, _atoms]
            feature = []
            for name in bond_feature:
                func = R.get("features.bond.%s" % name)
                feature += func(bond)
            _bond_feature += [feature, feature]
        edge_list = edge_list[:-2]
        bond_type = torch.tensor(bond_type)[:-2]
        bond_stereo = torch.tensor(bond_stereo)[:-2]
        stereo_atoms = torch.tensor(stereo_atoms)[:-2]
        if len(bond_feature) > 0:
            _bond_feature = torch.tensor(_bond_feature)[:-2]
        else:
            _bond_feature = None

        _mol_feature = []
        for name in mol_feature:
            func = R.get("features.molecule.%s" % name)
            _mol_feature += func(mol)
        if len(mol_feature) > 0:
            _mol_feature = torch.tensor(_mol_feature)
        else:
            _mol_feature = None

        num_relation = len(cls.bond2id) - 1 if kekulize else len(cls.bond2id)
        return cls(edge_list, atom_type, bond_type,
                   formal_charge=formal_charge, explicit_hs=explicit_hs,
                   chiral_tag=chiral_tag, radical_electrons=radical_electrons, atom_map=atom_map,
                   bond_stereo=bond_stereo, stereo_atoms=stereo_atoms, node_position=node_position,
                   atom_feature=_atom_feature, bond_feature=_bond_feature, mol_feature=_mol_feature,
                   num_node=mol.GetNumAtoms(), num_relation=num_relation)

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_smiles(cls, smiles, atom_feature="default", bond_feature="default", mol_feature=None,
                    with_hydrogen=False, kekulize=False):
        """
        Create a molecule from a SMILES string.

        Parameters:
            smiles (str): SMILES string
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            with_hydrogen (bool, optional): store hydrogens in the molecule graph.
                By default, hydrogens are dropped
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES `%s`" % smiles)

        return cls.from_molecule(mol, atom_feature, bond_feature, mol_feature, with_hydrogen, kekulize)

    def to_smiles(self, isomeric=True, atom_map=True, canonical=False):
        """
        Return a SMILES string of this molecule.

        Parameters:
            isomeric (bool, optional): keep isomeric information or not
            atom_map (bool, optional): keep atom mapping or not
            canonical (bool, optional): if true, return the canonical form of smiles

        Returns:
            str
        """
        mol = self.to_molecule()
        if not atom_map:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric)
        if canonical:
            smiles_set = set()
            while smiles not in smiles_set:
                smiles_set.add(smiles)
                mol = Chem.MolFromSmiles(smiles)
                smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric)
        return smiles

    def to_molecule(self, ignore_error=False):
        """
        Return an RDKit object of this molecule.

        Parameters:
            ignore_error (bool, optional): if true, return ``None`` for illegal molecules.
                Otherwise, raise an exception.

        Returns:
            rdchem.Mol
        """
        mol = Chem.RWMol()

        atom_type = self.atom_type.tolist()
        bond_type = self.bond_type.tolist()
        formal_charge = self.formal_charge.tolist()
        explicit_hs = self.explicit_hs.tolist()
        chiral_tag = self.chiral_tag.tolist()
        radical_electrons = self.radical_electrons.tolist()
        atom_map = self.atom_map.tolist()
        bond_stereo = self.bond_stereo.tolist()
        stereo_atoms = self.stereo_atoms.tolist()
        if hasattr(self, "node_position"):
            node_position = self.node_position.tolist()
            conformer = Chem.Conformer()
        else:
            conformer = None
        for i in range(self.num_node):
            atom = Chem.Atom(atom_type[i])
            atom.SetFormalCharge(formal_charge[i])
            atom.SetNumExplicitHs(explicit_hs[i])
            atom.SetChiralTag(Chem.ChiralType(chiral_tag[i]))
            atom.SetNumRadicalElectrons(radical_electrons[i])
            atom.SetNoImplicit(explicit_hs[i] > 0 or radical_electrons[i] > 0)
            atom.SetAtomMapNum(atom_map[i])
            if conformer:
                conformer.SetAtomPosition(i, node_position[i])
            mol.AddAtom(atom)
        if conformer:
            mol.AddConformer(conformer)

        edge_list = self.edge_list.tolist()
        for i in range(self.num_edge):
            h, t, type = edge_list[i]
            if h < t:
                j = mol.AddBond(h, t, Chem.BondType.names[self.id2bond[type]])
                bond = mol.GetBondWithIdx(j - 1)
                bond.SetIsAromatic(bond_type[i] == self.bond2id["AROMATIC"])
                bond.SetStereo(Chem.BondStereo(bond_stereo[i]))
        j = 0
        for i in range(self.num_edge):
            h, t, type = edge_list[i]
            if h < t:
                if bond_stereo[i]:
                    bond = mol.GetBondWithIdx(j)
                    bond.SetStereoAtoms(*stereo_atoms[i])
                j += 1

        if ignore_error:
            try:
                with utils.no_rdkit_log():
                    mol.UpdatePropertyCache()
                    Chem.AssignStereochemistry(mol)
                    mol.ClearComputedProps()
                    mol.UpdatePropertyCache()
            except:
                mol = None
        else:
            mol.UpdatePropertyCache()
            Chem.AssignStereochemistry(mol)
            mol.ClearComputedProps()
            mol.UpdatePropertyCache()

        return mol

    def ion_to_molecule(self):
        """
        Convert ions to molecules by adjusting hydrogens and electrons.

        Note [N+] will not be converted.
        """
        data_dict = self.data_dict

        formal_charge = data_dict.pop("formal_charge")
        explicit_hs = data_dict.pop("explicit_hs")
        radical_electrons = data_dict.pop("radical_electrons")
        pos_nitrogen = (self.atom_type == 7) & (self.explicit_valence > 3)
        formal_charge = pos_nitrogen.long()
        explicit_hs = torch.zeros_like(explicit_hs)
        radical_electrons = torch.zeros_like(radical_electrons)

        return type(self)(self.edge_list, edge_weight=self.edge_weight,
                          num_node=self.num_node, num_relation=self.num_relation,
                          formal_charge=formal_charge, explicit_hs=explicit_hs, radical_electrons=radical_electrons,
                          meta_dict=self.meta_dict, **data_dict)

    def to_scaffold(self, chirality=False):
        """
        Return a scaffold SMILES string of this molecule.

        Parameters:
            chirality (bool, optional): consider chirality in the scaffold or not

        Returns:
            str
        """
        smiles = self.to_smiles()
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles, includeChirality=chirality)
        return scaffold

    def node_mask(self, index, compact=False):
        self._check_no_stereo()
        return super(Molecule, self).node_mask(index, compact)

    def edge_mask(self, index):
        self._check_no_stereo()
        return super(Molecule, self).edge_mask(index)

    def undirected(self, add_inverse=False):
        if add_inverse:
            raise ValueError("Bonds are undirected relations, but `add_inverse` is specified")
        return super(Molecule, self).undirected(add_inverse)

    def atom(self):
        """
        Context manager for atom attributes.
        """
        return self.node()

    def bond(self):
        """
        Context manager for bond attributes.
        """
        return self.edge()

    def mol(self):
        """
        Context manager for molecule attributes.
        """
        return self.graph()

    def atom_reference(self):
        """
        Context manager for atom references.
        """
        return self.node_reference()

    def bond_reference(self):
        """
        Context manager for bond references.
        """
        return self.edge_reference()

    def mol_reference(self):
        """
        Context mangaer for molecule references.
        """
        return self.graph_reference()

    @property
    def num_node(self):
        return self.num_atom

    @num_node.setter
    def num_node(self, value):
        self.num_atom = value

    @property
    def num_edge(self):
        return self.num_bond

    @num_edge.setter
    def num_edge(self, value):
        self.num_bond = value

    atom2graph = Graph.node2graph
    bond2graph = Graph.edge2graph

    @property
    def node_feature(self):
        return self.atom_feature

    @node_feature.setter
    def node_feature(self, value):
        self.atom_feature = value

    @property
    def edge_feature(self):
        return self.bond_feature

    @edge_feature.setter
    def edge_feature(self, value):
        self.bond_feature = value

    @property
    def graph_feature(self):
        return self.mol_feature

    @graph_feature.setter
    def graph_feature(self, value):
        self.mol_feature = value

    @utils.cached_property
    def explicit_valence(self):
        bond2valence = torch.tensor(self.bond2valence, device=self.device)
        explicit_valence = scatter_add(bond2valence[self.edge_list[:, 2]], self.edge_list[:, 0], dim_size=self.num_node)
        return explicit_valence.round().long()

    @utils.cached_property
    def is_valid(self):
        """A coarse implementation of valence check."""
        # TODO: cross-check by any domain expert
        atom2valence = torch.tensor(float("nan")).repeat(constant.NUM_ATOM)
        for k, v in self.atom2valence:
            atom2valence[k] = v
        atom2valence = torch.as_tensor(atom2valence, device=self.device)

        max_atom_valence = atom2valence[self.atom_type]
        # special case for nitrogen
        pos_nitrogen = (self.atom_type == 7) & (self.formal_charge == 1)
        max_atom_valence[pos_nitrogen] = 4
        if torch.isnan(max_atom_valence).any():
            index = torch.isnan(max_atom_valence).nonzero()[0]
            raise ValueError("Fail to check valence. Unknown atom type %d" % self.atom_type[index])

        is_valid = (self.explicit_valence <= max_atom_valence).all()
        return is_valid

    @utils.cached_property
    def is_valid_rdkit(self):
        try:
            with utils.no_rdkit_log():
                mol = self.to_molecule()
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            is_valid = torch.ones(1, dtype=torch.bool, device=self.device)
        except ValueError:
            is_valid = torch.zeros(1, dtype=torch.bool, device=self.device)
        return is_valid

    def __repr__(self):
        fields = ["num_atom=%d" % self.num_atom, "num_bond=%d" % self.num_bond]
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))

    def visualize(self, title=None, save_file=None, figure_size=(3, 3), ax=None, atom_map=False):
        """
        Visualize this molecule with matplotlib.

        Parameters:
            title (str, optional): title for this molecule
            save_file (str, optional): ``png`` or ``pdf`` file to save visualization.
                If not provided, show the figure in window.
            figure_size (tuple of int, optional): width and height of the figure
            ax (matplotlib.axes.Axes, optional): axis to plot the figure
            atom_map (bool, optional): visualize atom mapping or not
        """
        is_root = ax is None
        if ax is None:
            fig = plt.figure(figsize=figure_size)
            if title is not None:
                ax = plt.gca()
            else:
                ax = fig.add_axes([0, 0, 1, 1])
        if title is not None:
            ax.set_title(title)

        mol = self.to_molecule()
        if not atom_map:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        draw.MolToMPL(mol, ax=ax)
        ax.set_frame_on(False)

        if is_root:
            if save_file:
                fig.savefig(save_file)
            else:
                fig.show()

    def __eq__(self, other):
        smiles = self.to_smiles(isomeric=False, atom_map=False, canonical=True)
        other_smiles = other.to_smiles(isomeric=False, atom_map=False, canonical=True)
        return smiles == other_smiles


class PackedMolecule(PackedGraph, Molecule):
    """
    Container for molecules with variadic sizes.

    .. warning::

        Edges of the same molecule are guaranteed to be consecutive in the edge list.
        However, this class doesn't enforce any order on the edges.

    Parameters:
        edge_list (array_like, optional): list of edges of shape :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out, bond_type).
        atom_type (array_like, optional): atom types of shape :math:`(|V|,)`
        bond_type (array_like, optional): bond types of shape :math:`(|E|,)`
        num_nodes (array_like, optional): number of nodes in each graph
            By default, it will be inferred from the largest id in `edge_list`
        num_edges (array_like, optional): number of edges in each graph
        offsets (array_like, optional): node id offsets of shape :math:`(|E|,)`.
            If not provided, nodes in `edge_list` should be relative index, i.e., the index in each graph.
            If provided, nodes in `edge_list` should be absolute index, i.e., the index in the packed graph.
    """

    unpacked_type = Molecule
    atom2graph = PackedGraph.node2graph
    bond2graph = PackedGraph.edge2graph

    def __init__(self, edge_list=None, atom_type=None, bond_type=None, num_nodes=None, num_edges=None, offsets=None,
                 **kwargs):
        if "num_relation" not in kwargs:
            kwargs["num_relation"] = len(self.bond2id)
        super(PackedMolecule, self).__init__(edge_list=edge_list, num_nodes=num_nodes, num_edges=num_edges,
                                             offsets=offsets, atom_type=atom_type, bond_type=bond_type, **kwargs)

    def ion_to_molecule(self):
        """
        Convert ions to molecules by adjusting hydrogens and electrons.

        Note [N+] will not be converted.
        """
        data_dict = self.data_dict

        formal_charge = data_dict.pop("formal_charge")
        explicit_hs = data_dict.pop("explicit_hs")
        radical_electrons = data_dict.pop("radical_electrons")
        pos_nitrogen = (self.atom_type == 7) & (self.explicit_valence > 3)
        formal_charge = pos_nitrogen.long()
        explicit_hs = torch.zeros_like(explicit_hs)
        radical_electrons = torch.zeros_like(radical_electrons)

        return type(self)(self.edge_list, edge_weight=self.edge_weight,
                          num_nodes=self.num_nodes, num_edges=self.num_edges, num_relation=self.num_relation,
                          offsets=self._offsets, formal_charge=formal_charge, explicit_hs=explicit_hs,
                          radical_electrons=radical_electrons, meta_dict=self.meta_dict, **data_dict)

    @utils.cached_property
    def is_valid(self):
        """A coarse implementation of valence check."""
        # TODO: cross-check by any domain expert
        atom2valence = torch.tensor(float("nan")).repeat(constant.NUM_ATOM)
        for k, v in self.atom2valence.items():
            atom2valence[k] = v
        atom2valence = torch.as_tensor(atom2valence, device=self.device)

        max_atom_valence = atom2valence[self.atom_type]
        # special case for nitrogen
        pos_nitrogen = (self.atom_type == 7) & (self.formal_charge == 1)
        max_atom_valence[pos_nitrogen] = 4
        if torch.isnan(max_atom_valence).any():
            index = torch.isnan(max_atom_valence).nonzero()[0]
            raise ValueError("Fail to check valence. Unknown atom type %d" % self.atom_type[index])

        is_valid = self.explicit_valence <= max_atom_valence
        is_valid = scatter_min(is_valid.long(), self.node2graph, dim_size=self.batch_size)[0].bool()
        return is_valid

    @utils.cached_property
    def is_valid_rdkit(self):
        return torch.cat([mol.is_valid_rdkit for mol in self])

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_molecule(cls, mols, atom_feature="default", bond_feature="default", mol_feature=None,
                      with_hydrogen=False, kekulize=False):
        """
        Create a packed molecule from a list of RDKit objects.

        Parameters:
            mols (list of rdchem.Mol): molecules
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            with_hydrogen (bool, optional): store hydrogens in the molecule graph.
                By default, hydrogens are dropped
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        atom_feature = cls._standarize_option(atom_feature)
        bond_feature = cls._standarize_option(bond_feature)
        mol_feature = cls._standarize_option(mol_feature)

        atom_type = []
        formal_charge = []
        explicit_hs = []
        chiral_tag = []
        radical_electrons = []
        atom_map = []

        edge_list = []
        bond_type = []
        bond_stereo = []
        stereo_atoms = []
        node_position = []

        _atom_feature = []
        _bond_feature = []
        _mol_feature = []
        num_nodes = []
        num_edges = []

        mols = mols + [cls.dummy_mol]
        for mol in mols:
            if mol is None:
                mol = cls.empty_mol
            # some RDKit operations are in-place
            # copy the object to avoid undesired behavior in the caller
            mol = copy(mol)
            if with_hydrogen:
                mol = Chem.AddHs(mol)
            if kekulize:
                Chem.Kekulize(mol)

            if mol.GetNumConformers() > 0:
                node_position += mol.GetConformer().GetPositions().tolist()
            for atom in mol.GetAtoms():
                atom_type.append(atom.GetAtomicNum())
                formal_charge.append(atom.GetFormalCharge())
                explicit_hs.append(atom.GetNumExplicitHs())
                chiral_tag.append(atom.GetChiralTag())
                radical_electrons.append(atom.GetNumRadicalElectrons())
                atom_map.append(atom.GetAtomMapNum())
                feature = []
                for name in atom_feature:
                    func = R.get("features.atom.%s" % name)
                    feature += func(atom)
                _atom_feature.append(feature)

            for bond in mol.GetBonds():
                type = str(bond.GetBondType())
                stereo = bond.GetStereo()
                if stereo:
                    _atoms = list(bond.GetStereoAtoms())
                else:
                    _atoms = [0, 0]
                if type not in cls.bond2id:
                    continue
                type = cls.bond2id[type]
                h, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                feature = []
                for name in bond_feature:
                    func = R.get("features.bond.%s" % name)
                    feature += func(bond)
                edge_list += [[h, t, type], [t, h, type]]
                # always explicitly store aromatic bonds
                if bond.GetIsAromatic():
                    type = cls.bond2id["AROMATIC"]
                bond_type += [type, type]
                bond_stereo += [stereo, stereo]
                stereo_atoms += [_atoms, _atoms]
                _bond_feature += [feature, feature]

            feature = []
            for name in mol_feature:
                func = R.get("features.molecule.%s" % name)
                feature += func(mol)
            _mol_feature.append(feature)

            num_nodes.append(mol.GetNumAtoms())
            num_edges.append(mol.GetNumBonds() * 2)

        atom_type = torch.tensor(atom_type)[:-2]
        atom_map = torch.tensor(atom_map)[:-2]
        formal_charge = torch.tensor(formal_charge)[:-2]
        explicit_hs = torch.tensor(explicit_hs)[:-2]
        chiral_tag = torch.tensor(chiral_tag)[:-2]
        radical_electrons = torch.tensor(radical_electrons)[:-2]
        if len(node_position) > 0:
            node_position = torch.tensor(node_position)
        else:
            node_position = None
        if len(atom_feature) > 0:
            _atom_feature = torch.tensor(_atom_feature)[:-2]
        else:
            _atom_feature = None

        num_nodes = num_nodes[:-1]
        num_edges = num_edges[:-1]
        edge_list = torch.tensor(edge_list)[:-2]
        bond_type = torch.tensor(bond_type)[:-2]
        bond_stereo = torch.tensor(bond_stereo)[:-2]
        stereo_atoms = torch.tensor(stereo_atoms)[:-2]
        if len(bond_feature) > 0:
            _bond_feature = torch.tensor(_bond_feature)[:-2]
        else:
            _bond_feature = None
        if len(mol_feature) > 0:
            _mol_feature = torch.tensor(_mol_feature)[:-1]
        else:
            _mol_feature = None

        num_relation = len(cls.bond2id) - 1 if kekulize else len(cls.bond2id)
        return cls(edge_list, atom_type, bond_type,
                   formal_charge=formal_charge, explicit_hs=explicit_hs,
                   chiral_tag=chiral_tag, radical_electrons=radical_electrons, atom_map=atom_map,
                   bond_stereo=bond_stereo, stereo_atoms=stereo_atoms, node_position=node_position,
                   atom_feature=_atom_feature, bond_feature=_bond_feature, mol_feature=_mol_feature,
                   num_nodes=num_nodes, num_edges=num_edges, num_relation=num_relation)

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_smiles(cls, smiles_list, atom_feature="default", bond_feature="default", mol_feature=None,
                    with_hydrogen=False, kekulize=False):
        """
        Create a packed molecule from a list of SMILES strings.

        Parameters:
            smiles_list (str): list of SMILES strings
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            with_hydrogen (bool, optional): store hydrogens in the molecule graph.
                By default, hydrogens are dropped
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        mols = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES `%s`" % smiles)
            mols.append(mol)

        return cls.from_molecule(mols, atom_feature, bond_feature, mol_feature, with_hydrogen, kekulize)

    def to_smiles(self, isomeric=True, atom_map=True, canonical=False):
        """
        Return a list of SMILES strings.

        Parameters:
            isomeric (bool, optional): keep isomeric information or not
            atom_map (bool, optional): keep atom mapping or not
            canonical (bool, optional): if true, return the canonical form of smiles

        Returns:
            list of str
        """
        mols = self.to_molecule()
        smiles_list = []
        for mol in mols:
            if not atom_map:
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric)
            if canonical:
                smiles_set = set()
                while smiles not in smiles_set:
                    smiles_set.add(smiles)
                    mol = Chem.MolFromSmiles(smiles)
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric)
            smiles_list.append(smiles)
        return smiles_list

    def to_molecule(self, ignore_error=False):
        """
        Return a list of RDKit objects.

        Parameters:
            ignore_error (bool, optional): if true, return ``None`` for illegal molecules.
                Otherwise, raise an exception.

        Returns:
            list of rdchem.Mol
        """
        atom_type = self.atom_type.tolist()
        bond_type = self.bond_type.tolist()
        formal_charge = self.formal_charge.tolist()
        explicit_hs = self.explicit_hs.tolist()
        chiral_tag = self.chiral_tag.tolist()
        radical_electrons = self.radical_electrons.tolist()
        atom_map = self.atom_map.tolist()
        bond_stereo = self.bond_stereo.tolist()
        stereo_atoms = self.stereo_atoms.tolist()
        if hasattr(self, "node_position"):
            node_position = self.node_position.tolist()
        else:
            node_position = None
        num_cum_nodes = [0] + self.num_cum_nodes.tolist()
        num_cum_edges = [0] + self.num_cum_edges.tolist()
        edge_list = self.edge_list.clone()
        edge_list[:, :2] -= self._offsets.unsqueeze(-1)
        edge_list = edge_list.tolist()

        mols = []
        for i in range(self.batch_size):
            mol = Chem.RWMol()
            if node_position:
                conformer = Chem.Conformer()
            else:
                conformer = None
            for j in range(num_cum_nodes[i], num_cum_nodes[i + 1]):
                atom = Chem.Atom(atom_type[j])
                atom.SetFormalCharge(formal_charge[j])
                atom.SetNumExplicitHs(explicit_hs[j])
                atom.SetChiralTag(Chem.ChiralType(chiral_tag[j]))
                atom.SetNumRadicalElectrons(radical_electrons[j])
                atom.SetNoImplicit(explicit_hs[j] > 0 or radical_electrons[j] > 0)
                atom.SetAtomMapNum(atom_map[j])
                if conformer:
                    conformer.SetAtomPosition(j - num_cum_nodes[i], node_position[j])
                mol.AddAtom(atom)
            if conformer:
                mol.AddConformer(conformer)

            for j in range(num_cum_edges[i], num_cum_edges[i + 1]):
                h, t, type = edge_list[j]
                if h < t:
                    k = mol.AddBond(h, t, Chem.BondType.names[self.id2bond[type]])
                    bond = mol.GetBondWithIdx(k - 1)
                    bond.SetIsAromatic(bond_type[j] == self.bond2id["AROMATIC"])
                    bond.SetStereo(Chem.BondStereo(bond_stereo[j]))
            k = 0
            for j in range(num_cum_edges[i], num_cum_edges[i + 1]):
                h, t, type = edge_list[j]
                if h < t:
                    if bond_stereo[j]:
                        bond = mol.GetBondWithIdx(k)
                        # These do not necessarily need to be the highest 'ranking' atoms like CIP stereo requires. 
                        # They can be any arbitrary atoms neighboring the begin and end atoms of this bond respectively. 
                        # STEREOCIS or STEREOTRANS is then set relative to only these atoms.
                        bond.SetStereoAtoms(*stereo_atoms[j])
                    k += 1

            if ignore_error:
                try:
                    with utils.no_rdkit_log():
                        mol.UpdatePropertyCache()
                        Chem.AssignStereochemistry(mol)
                        mol.ClearComputedProps()
                        mol.UpdatePropertyCache()
                except:
                    mol = None
            else:
                mol.UpdatePropertyCache()
                Chem.AssignStereochemistry(mol)
                mol.ClearComputedProps()
                mol.UpdatePropertyCache()
            mols.append(mol)

        return mols

    def node_mask(self, index, compact=False):
        self._check_no_stereo()
        return super(PackedMolecule, self).node_mask(index, compact)

    def edge_mask(self, index):
        self._check_no_stereo()
        return super(PackedMolecule, self).edge_mask(index)

    def undirected(self, add_inverse=False):
        if add_inverse:
            raise ValueError("Bonds are undirected relations, but `add_inverse` is specified")
        return super(PackedMolecule, self).undirected(add_inverse)

    @property
    def num_nodes(self):
        return self.num_atoms

    @num_nodes.setter
    def num_nodes(self, value):
        self.num_atoms = value

    @property
    def num_edges(self):
        return self.num_bonds

    @num_edges.setter
    def num_edges(self, value):
        self.num_bonds = value

    def __repr__(self):
        fields = ["batch_size=%d" % self.batch_size,
                  "num_atoms=%s" % pretty.long_array(self.num_atoms.tolist()),
                  "num_bonds=%s" % pretty.long_array(self.num_bonds.tolist())]
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))

    def visualize(self, titles=None, save_file=None, figure_size=(3, 3), num_row=None, num_col=None, atom_map=False):
        """
        Visualize the packed molecules with matplotlib.

        Parameters:
            titles (list of str, optional): title for each molecule. Default is the ID of each molecule.
            save_file (str, optional): ``png`` or ``pdf`` file to save visualization.
                If not provided, show the figure in window.
            figure_size (tuple of int, optional): width and height of the figure
            num_row (int, optional): number of rows in the figure
            num_col (int, optional): number of columns in the figure
            atom_map (bool, optional): visualize atom mapping or not
        """
        if titles is None:
            graph = self.get_item(0)
            titles = ["%s %d" % (type(graph).__name__, i) for i in range(self.batch_size)]
        if num_col is None:
            if num_row is None:
                num_col = math.ceil(self.batch_size ** 0.5)
            else:
                num_col = math.ceil(self.batch_size / num_row)
        if num_row is None:
            num_row = math.ceil(self.batch_size / num_col)

        figure_size = (num_col * figure_size[0], num_row * figure_size[1])
        fig = plt.figure(figsize=figure_size)

        for i in range(self.batch_size):
            graph = self.get_item(i)
            ax = fig.add_subplot(num_row, num_col, i + 1)
            graph.visualize(title=titles[i], ax=ax, atom_map=atom_map)
        # remove the space of axis labels
        fig.tight_layout()

        if save_file:
            fig.savefig(save_file)
        else:
            fig.show()


Molecule.packed_type = PackedMolecule