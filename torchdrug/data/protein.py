import os
import warnings
from collections import defaultdict

from rdkit import Chem
import torch
from torch_scatter import scatter_add, scatter_max, scatter_min

from torchdrug import utils
from torchdrug.data import Molecule, PackedMolecule, Dictionary, feature
from torchdrug.core import Registry as R
from torchdrug.utils import pretty


class Protein(Molecule):
    """
    Proteins with predefined chemical features.
    Support both residue-level and atom-level operations and ensure consistency between two views.

    .. warning::

        The order of residues must be the same as the protein sequence.
        However, this class doesn't enforce any order on nodes or edges.
        Nodes may have a different order with residues.

    Parameters:
        edge_list (array_like, optional): list of edges of shape :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out, bond_type).
        atom_type (array_like, optional): atom types of shape :math:`(|V|,)`
        bond_type (array_like, optional): bond types of shape :math:`(|E|,)`
        residue_type (array_like, optional): residue types of shape :math:`(|V_{res}|,)`
        view (str, optional): default view for this protein. Can be ``atom`` or ``residue``.
        atom_name (array_like, optional): atom names in a residue of shape :math:`(|V|,)`
        atom2residue (array_like, optional): atom id to residue id mapping of shape :math:`(|V|,)`
        residue_feature (array_like, optional): residue features of shape :math:`(|V_{res}|, ...)`
        is_hetero_atom (array_like, optional): hetero atom indicators of shape :math:`(|V|,)`
        occupancy (array_like, optional): occupancy of shape :math:`(|V|,)`
        b_factor (array_like, optional): temperature factors of shape :math:`(|V|,)`
        residue_number (array_like, optional): residue numbers of shape :math:`(|V_{res}|,)`
        insertion_code (array_like, optional): insertion codes of shape :math:`(|V_{res}|,)`
        chain_id (array_like, optional): chain ids of shape :math:`(|V_{res}|,)`
    """

    _meta_types = {"node", "edge", "residue", "graph",
                   "node reference", "edge reference", "residue reference", "graph reference"}
    dummy_protein = Chem.MolFromSequence("G")
    dummy_atom = dummy_protein.GetAtomWithIdx(0)

    # TODO: rdkit isn't compatible with X in the sequence
    residue2id = {"GLY": 0, "ALA": 1, "SER": 2, "PRO": 3, "VAL": 4, "THR": 5, "CYS": 6, "ILE": 7, "LEU": 8,
                  "ASN": 9, "ASP": 10, "GLN": 11, "LYS": 12, "GLU": 13, "MET": 14, "HIS": 15, "PHE": 16,
                  "ARG": 17, "TYR": 18, "TRP": 19}
    atom_name2id = {"C": 0, "CA": 1, "CB": 2, "CD": 3, "CD1": 4, "CD2": 5, "CE": 6, "CE1": 7, "CE2": 8,
                    "CE3": 9, "CG": 10, "CG1": 11, "CG2": 12, "CH2": 13, "CZ": 14, "CZ2": 15, "CZ3": 16,
                    "N": 17, "ND1": 18, "ND2": 19, "NE": 20, "NE1": 21, "NE2": 22, "NH1": 23, "NH2": 24,
                    "NZ": 25, "O": 26, "OD1": 27, "OD2": 28, "OE1": 29, "OE2": 30, "OG": 31, "OG1": 32,
                    "OH": 33, "OXT": 34, "SD": 35, "SG": 36, "UNK": 37}
    alphabet2id = {" ": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10,
                   "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20,
                   "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26}
    id2residue = {v: k for k, v in residue2id.items()}
    id2residue_symbol = {0: "G", 1: "A", 2: "S", 3: "P", 4: "V", 5: "T", 6: "C", 7: "I", 8: "L", 9: "N",
                         10: "D", 11: "Q", 12: "K", 13: "E", 14: "M", 15: "H", 16: "F", 17: "R", 18: "Y", 19: "W"}
    residue_symbol2id = {v: k for k, v in id2residue_symbol.items()}
    id2atom_name = {v: k for k, v in atom_name2id.items()}
    id2alphabet = {v: k for k, v in alphabet2id.items()}

    def __init__(self, edge_list=None, atom_type=None, bond_type=None, residue_type=None, view=None,
                 atom_name=None, atom2residue=None, residue_feature=None, is_hetero_atom=None, occupancy=None,
                 b_factor=None, residue_number=None, insertion_code=None, chain_id=None, **kwargs):
        super(Protein, self).__init__(edge_list, atom_type, bond_type, **kwargs)
        residue_type, num_residue = self._standarize_num_residue(residue_type)
        self.num_residue = num_residue
        self.view = self._standarize_view(view)

        atom_name = self._standarize_attribute(atom_name, self.num_node)
        atom2residue = self._standarize_attribute(atom2residue, self.num_node)
        is_hetero_atom = self._standarize_attribute(is_hetero_atom, self.num_node, dtype=torch.bool)
        occupancy = self._standarize_attribute(occupancy, self.num_node, dtype=torch.float, default=1)
        b_factor = self._standarize_attribute(b_factor, self.num_node, dtype=torch.float)
        residue_number = self._standarize_attribute(residue_number, self.num_residue)
        insertion_code = self._standarize_attribute(insertion_code, self.num_residue)
        chain_id = self._standarize_attribute(chain_id, self.num_residue)

        with self.atom():
            self.atom_name = atom_name
            with self.residue_reference():
                self.atom2residue = atom2residue
            self.is_hetero_atom = is_hetero_atom
            self.occupancy = occupancy
            self.b_factor = b_factor

        with self.residue():
            self.residue_type = residue_type
            if residue_feature is not None:
                self.residue_feature = torch.as_tensor(residue_feature, device=self.device)
            self.residue_number = residue_number
            self.insertion_code = insertion_code
            self.chain_id = chain_id

    def residue(self):
        """
        Context manager for residue attributes.
        """
        return self.context("residue")

    def residue_reference(self):
        """
        Context manager for residue references.
        """
        return self.context("residue reference")

    @property
    def node_feature(self):
        if getattr(self, "view", "atom") == "atom":
            return self.atom_feature
        else:
            return self.residue_feature

    @node_feature.setter
    def node_feature(self, value):
        self.atom_feature = value

    @property
    def num_node(self):
        return self.num_atom

    @num_node.setter
    def num_node(self, value):
        self.num_atom = value

    def _check_attribute(self, key, value):
        super(Protein, self)._check_attribute(key, value)
        for type in self._meta_contexts:
            if type == "residue":
                if len(value) != self.num_residue:
                    raise ValueError("Expect residue attribute `%s` to have shape (%d, *), but found %s" %
                                     (key, self.num_residue, value.shape))
            elif type == "residue reference":
                is_valid = (value >= -1) & (value < self.num_residue)
                if not is_valid.all():
                    error_value = value[~is_valid]
                    raise ValueError("Expect residue reference in [-1, %d), but found %d" %
                                     (self.num_residue, error_value[0]))

    def _standarize_num_residue(self, residue_type):
        if residue_type is None:
            raise ValueError("`residue_type` should be provided")

        residue_type = torch.as_tensor(residue_type, dtype=torch.long, device=self.device)
        num_residue = torch.tensor(len(residue_type), device=self.device)
        return residue_type, num_residue

    def __setattr__(self, key, value):
        if key == "view" and value not in ["atom", "residue"]:
            raise ValueError("Expect `view` to be either `atom` or `residue`, but found `%s`" % value)
        return super(Protein, self).__setattr__(key, value)

    def _standarize_view(self, view):
        if view is None:
            if self.num_atom > 0:
                view = "atom"
            else:
                view = "residue"
        return view

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_molecule(cls, mol, atom_feature="default", bond_feature="default", residue_feature="default",
                      mol_feature=None, kekulize=False):
        """
        Create a protein from an RDKit object.

        Parameters:
            mol (rdchem.Mol): molecule
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str, list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        protein = Molecule.from_molecule(mol, atom_feature=atom_feature, bond_feature=bond_feature,
                                         mol_feature=mol_feature, with_hydrogen=False, kekulize=kekulize)
        residue_feature = cls._standarize_option(residue_feature)

        if kekulize:
            Chem.Kekulize(mol)

        residue_type = []
        atom_name = []
        is_hetero_atom = []
        occupancy = []
        b_factor = []
        atom2residue = []
        residue_number = []
        insertion_code = []
        chain_id = []
        _residue_feature = []
        last_residue = None
        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())] + [cls.dummy_atom]
        for atom in atoms:
            pdbinfo = atom.GetPDBResidueInfo()
            number = pdbinfo.GetResidueNumber()
            code = pdbinfo.GetInsertionCode()
            type = pdbinfo.GetResidueName().strip()
            canonical_residue = (number, code, type)
            if canonical_residue != last_residue:
                last_residue = canonical_residue
                if type not in cls.residue2id:
                    warnings.warn("Unknown residue `%s`. Treat as glycine" % type)
                    type = "GLY"
                residue_type.append(cls.residue2id[type])
                residue_number.append(number)
                if pdbinfo.GetInsertionCode() not in cls.alphabet2id or pdbinfo.GetChainId() not in cls.alphabet2id:
                    return None
                insertion_code.append(cls.alphabet2id[pdbinfo.GetInsertionCode()])
                chain_id.append(cls.alphabet2id[pdbinfo.GetChainId()])
                feature = []
                for name in residue_feature:
                    func = R.get("features.residue.%s" % name)
                    feature += func(pdbinfo)
                _residue_feature.append(feature)
            name = pdbinfo.GetName().strip()
            if name not in cls.atom_name2id:
                name = "UNK"
            atom_name.append(cls.atom_name2id[name])
            is_hetero_atom.append(pdbinfo.GetIsHeteroAtom())
            occupancy.append(pdbinfo.GetOccupancy())
            b_factor.append(pdbinfo.GetTempFactor())
            atom2residue.append(len(residue_type) - 1)
        residue_type = torch.tensor(residue_type)[:-1]
        atom_name = torch.tensor(atom_name)[:-1]
        is_hetero_atom = torch.tensor(is_hetero_atom)[:-1]
        occupancy = torch.tensor(occupancy)[:-1]
        b_factor = torch.tensor(b_factor)[:-1]
        atom2residue = torch.tensor(atom2residue)[:-1]
        residue_number = torch.tensor(residue_number)[:-1]
        insertion_code = torch.tensor(insertion_code)[:-1]
        chain_id = torch.tensor(chain_id)[:-1]
        if len(residue_feature) > 0:
            _residue_feature = torch.tensor(_residue_feature)[:-1]
        else:
            _residue_feature = None

        return cls(protein.edge_list, num_node=protein.num_node, residue_type=residue_type,
                   atom_name=atom_name, atom2residue=atom2residue, residue_feature=_residue_feature,
                   is_hetero_atom=is_hetero_atom, occupancy=occupancy, b_factor=b_factor,
                   residue_number=residue_number, insertion_code=insertion_code, chain_id=chain_id,
                   meta_dict=protein.meta_dict, **protein.data_dict)

    @classmethod
    def _residue_from_sequence(cls, sequence):
        residue_type = []
        residue_feature = []
        sequence = sequence + "G"
        for residue in sequence:
            if residue not in cls.residue_symbol2id:
                warnings.warn("Unknown residue symbol `%s`. Treat as glycine" % residue)
                residue = "G"
            residue_type.append(cls.residue_symbol2id[residue])
            residue_feature.append(feature.onehot(residue, cls.residue_symbol2id, allow_unknown=True))

        residue_type = residue_type[:-1]
        residue_feature = torch.tensor(residue_feature)[:-1]

        return cls(edge_list=None, atom_type=[], bond_type=[], num_node=0, residue_type=residue_type,
                   residue_feature=residue_feature)

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_sequence(cls, sequence, atom_feature="default", bond_feature="default", residue_feature="default",
                      mol_feature=None, kekulize=False):
        """
        Create a protein from a sequence.

        .. note::

            It takes considerable time to construct proteins with a large number of atoms and bonds.
            If you only need residue information, you may speed up the construction by setting
            ``atom_feature`` and ``bond_feature`` to ``None``.

        Parameters:
            sequence (str): protein sequence
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str, list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if atom_feature is None and bond_feature is None and residue_feature == "default":
            return cls._residue_from_sequence(sequence)
        
        mol = Chem.MolFromSequence(sequence)
        if mol is None:
            raise ValueError("Invalid sequence `%s`" % sequence)

        return cls.from_molecule(mol, atom_feature, bond_feature, residue_feature, mol_feature, kekulize)

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_pdb(cls, pdb_file, atom_feature="default", bond_feature="default", residue_feature="default",
                 mol_feature=None, kekulize=False):
        """
        Create a protein from a PDB file.

        Parameters:
            pdb_file (str): file name
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str, list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if not os.path.exists(pdb_file):
            raise FileNotFoundError("No such file `%s`" % pdb_file)
        mol = Chem.MolFromPDBFile(pdb_file)
        if mol is None:
            raise ValueError("RDKit cannot read PDB file `%s`" % pdb_file)
        return cls.from_molecule(mol, atom_feature, bond_feature, residue_feature, mol_feature, kekulize)

    def to_molecule(self, ignore_error=False):
        """
        Return an RDKit object of this protein.

        Parameters:
            ignore_error (bool, optional): if true, return ``None`` for illegal molecules.
                Otherwise, raise an exception.

        Returns:
            rdchem.Mol
        """
        mol = super(Protein, self).to_molecule(ignore_error)
        if mol is None:
            return mol

        residue_type = self.residue_type.tolist()
        atom_name = self.atom_name.tolist()
        atom2residue = self.atom2residue.tolist()
        is_hetero_atom = self.is_hetero_atom.tolist()
        occupancy = self.occupancy.tolist()
        b_factor = self.b_factor.tolist()
        residue_number = self.residue_number.tolist()
        chain_id = self.chain_id.tolist()
        insertion_code = self.insertion_code.tolist()
        for i, atom in enumerate(mol.GetAtoms()):
            r = atom2residue[i]
            residue = Chem.AtomPDBResidueInfo()
            residue.SetResidueNumber(residue_number[r])
            residue.SetChainId(self.id2alphabet[chain_id[r]])
            residue.SetInsertionCode(self.id2alphabet[insertion_code[r]])
            residue.SetName(" %-3s" % self.id2atom_name[atom_name[i]])
            residue.SetResidueName(self.id2residue[residue_type[r]])
            residue.SetIsHeteroAtom(is_hetero_atom[i])
            residue.SetOccupancy(occupancy[i])
            residue.SetTempFactor(b_factor[i])
            atom.SetPDBResidueInfo(residue)

        return mol

    def to_sequence(self):
        """
        Return a sequence of this protein.

        Returns:
            str
        """
        residue_type = self.residue_type.tolist()
        cc_id = self.connected_component_id.tolist()
        sequence = []
        for i in range(self.num_residue):
            if i > 0 and cc_id[i] > cc_id[i - 1]:
                sequence.append(".")
            sequence.append(self.id2residue_symbol[residue_type[i]])
        return "".join(sequence)

    def to_pdb(self, pdb_file):
        """
        Write this protein to a pdb file.

        Parameters:
            pdb_file (str): file name
        """
        mol = self.to_molecule()
        Chem.MolToPDBFile(mol, pdb_file, flavor=10)

    def split(self, node2graph):
        node2graph = torch.as_tensor(node2graph, dtype=torch.long, device=self.device)
        # coalesce arbitrary graph IDs to [0, n)
        _, node2graph = torch.unique(node2graph, return_inverse=True)
        num_graph = node2graph.max() + 1
        index = node2graph.argsort()
        mapping = torch.zeros_like(index)
        mapping[index] = torch.arange(len(index), device=self.device)

        node_in, node_out = self.edge_list.t()[:2]
        edge_mask = node2graph[node_in] == node2graph[node_out]
        edge2graph = node2graph[node_in]
        edge_index = edge2graph.argsort()
        edge_index = edge_index[edge_mask[edge_index]]

        prepend = torch.tensor([-1], device=self.device)
        is_first_node = torch.diff(node2graph[index], prepend=prepend) > 0
        graph_index = self.node2graph[index[is_first_node]]

        # a residue can be split into multiple graphs
        max_num_node = node2graph.bincount(minlength=num_graph).max()
        key = node2graph[index] * max_num_node + self.atom2residue[index]
        key_set, atom2residue = key.unique(return_inverse=True)
        residue_index = key_set % max_num_node

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]

        num_nodes = node2graph.bincount(minlength=num_graph)
        num_edges = edge2graph[edge_index].bincount(minlength=num_graph)
        num_cum_residues = scatter_max(atom2residue, node2graph[index], dim_size=num_graph)[0] + 1
        prepend = torch.tensor([0], device=self.device)
        num_residues = torch.diff(num_cum_residues, prepend=prepend)

        num_cum_nodes = num_nodes.cumsum(0)
        offsets = (num_cum_nodes - num_nodes)[edge2graph[edge_index]]

        data_dict, meta_dict = self.data_mask(index, edge_index, residue_index, graph_index,
                                              exclude=("residue reference", "graph reference"))

        return self.packed_type(edge_list[edge_index], edge_weight=self.edge_weight[edge_index],
                                num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues, view=self.view,
                                offsets=offsets, atom2residue=atom2residue, meta_dict=meta_dict, **data_dict)

    @classmethod
    def pack(cls, graphs):
        edge_list = []
        edge_weight = []
        num_nodes = []
        num_edges = []
        num_residues = []
        num_cum_node = 0
        num_cum_edge = 0
        num_cum_residue = 0
        num_graph = 0
        data_dict = defaultdict(list)
        meta_dict = graphs[0].meta_dict
        view = graphs[0].view
        for graph in graphs:
            edge_list.append(graph.edge_list)
            edge_weight.append(graph.edge_weight)
            num_nodes.append(graph.num_node)
            num_edges.append(graph.num_edge)
            num_residues.append(graph.num_residue)
            for k, v in graph.data_dict.items():
                for type in meta_dict[k]:
                    if type == "graph":
                        v = v.unsqueeze(0)
                    elif type == "node reference":
                        v = torch.where(v != -1, v + num_cum_node, -1)
                    elif type == "edge reference":
                        v = torch.where(v != -1, v + num_cum_edge, -1)
                    elif type == "residue reference":
                        v = torch.where(v != -1, v + num_cum_residue, -1)
                    elif type == "graph reference":
                        v = torch.where(v != -1, v + num_graph, -1)
                data_dict[k].append(v)
            num_cum_node += graph.num_node
            num_cum_edge += graph.num_edge
            num_cum_residue += graph.num_residue
            num_graph += 1

        edge_list = torch.cat(edge_list)
        edge_weight = torch.cat(edge_weight)
        data_dict = {k: torch.cat(v) for k, v in data_dict.items()}

        return cls.packed_type(edge_list, edge_weight=edge_weight, num_relation=graphs[0].num_relation,
                               num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues, view=view,
                               meta_dict=meta_dict, **data_dict)

    def repeat(self, count):
        edge_list = self.edge_list.repeat(count, 1)
        edge_weight = self.edge_weight.repeat(count)
        num_nodes = [self.num_node] * count
        num_edges = [self.num_edge] * count
        num_residues = [self.num_residue] * count
        num_relation = self.num_relation

        data_dict = {}
        for k, v in self.data_dict.items():
            if "graph" in self.meta_dict[k]:
                v = v.unsqueeze(0)
            shape = [1] * v.ndim
            shape[0] = count
            length = len(v)
            v = v.repeat(shape)
            for type in self.meta_dict[k]:
                if type == "node reference":
                    offsets = torch.arange(count, device=self.device) * self.num_node
                    v = v + offsets.repeat_interleave(length)
                elif type == "edge reference":
                    offsets = torch.arange(count, device=self.device) * self.num_edge
                    v = v + offsets.repeat_interleave(length)
                elif type == "residue reference":
                    offsets = torch.arange(count, device=self.device) * self.num_residue
                    v = v + offsets.repeat_interleave(length)
                elif type == "graph reference":
                    offsets = torch.arange(count, device=self.device)
                    v = v + offsets.repeat_interleave(length)
            data_dict[k] = v

        return self.packed_type(edge_list, edge_weight=edge_weight,
                                num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues, view=self.view,
                                num_relation=num_relation, meta_dict=self.meta_dict, **data_dict)

    def residue2atom(self, residue_index):
        """Map residue ids to atom ids."""
        residue_index = self._standarize_index(residue_index, self.num_residue)
        if not hasattr(self, "node_inverted_index"):
            self.node_inverted_index = self._build_node_inverted_index()
        inverted_range, order = self.node_inverted_index
        starts, ends = inverted_range[residue_index].t()
        num_match = ends - starts
        offsets = num_match.cumsum(0) - num_match
        ranges = torch.arange(num_match.sum(), device=self.device)
        ranges = ranges + (starts - offsets).repeat_interleave(num_match)
        index = order[ranges]
        return index

    def _build_node_inverted_index(self):
        keys = self.atom2residue
        order = keys.argsort()
        keys_set, num_keys = keys.unique(return_counts=True)
        ends = num_keys.cumsum(0)
        starts = ends - num_keys
        ranges = torch.stack([starts, ends], dim=-1)
        inverted_range = Dictionary(keys_set, ranges)
        return inverted_range, order

    def __getitem__(self, index):
        # why do we check tuple?
        # case 1: x[0, 1] is parsed as (0, 1)
        # case 2: x[[0, 1]] is parsed as [0, 1]
        if not isinstance(index, tuple):
            index = (index,)

        if len(index) > 1:
            raise ValueError("Protein has only 1 axis, but %d axis is indexed" % len(index))

        return self.residue_mask(index[0], compact=True)

    def data_mask(self, node_index=None, edge_index=None, residue_index=None, graph_index=None, include=None,
                  exclude=None):
        data_dict, meta_dict = super(Protein, self).data_mask(node_index, edge_index, graph_index=graph_index,
                                                              include=include, exclude=exclude)
        residue_mapping = None
        for k, v in data_dict.items():
            for type in meta_dict[k]:
                if type == "residue" and residue_index is not None:
                    if v.is_sparse:
                        v = v.to_dense()[residue_index].to_sparse()
                    else:
                        v = v[residue_index]
                elif type == "residue reference" and residue_index is not None:
                    if residue_mapping is None:
                        residue_mapping = self._get_mapping(residue_index, self.num_residue)
                    v = residue_mapping[v]
            data_dict[k] = v

        return data_dict, meta_dict

    def residue_mask(self, index, compact=False):
        """
        Return a masked protein based on the specified residues.

        Note the compact option is applied to both residue and atom ids.

        Parameters:
            index (array_like): residue index
            compact (bool, optional): compact residue ids or not

        Returns:
            Protein
        """
        index = self._standarize_index(index, self.num_residue)
        if (torch.diff(index) <= 0).any():
            warnings.warn("`residue_mask()` is called to re-order the residues. This will change the protein sequence. "
                          "If this is not desired, you might have passed a wrong index to this function.")
        residue_mapping = -torch.ones(self.num_residue, dtype=torch.long, device=self.device)
        residue_mapping[index] = torch.arange(len(index), device=self.device)

        node_index = residue_mapping[self.atom2residue] >= 0
        node_index = self._standarize_index(node_index, self.num_node)
        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
        if compact:
            mapping[node_index] = torch.arange(len(node_index), device=self.device)
            num_node = len(node_index)
        else:
            mapping[node_index] = node_index
            num_node = self.num_node

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1)
        edge_index = self._standarize_index(edge_index, self.num_edge)

        if compact:
            data_dict, meta_dict = self.data_mask(node_index, edge_index, residue_index=index)
        else:
            data_dict, meta_dict = self.data_mask(edge_index=edge_index)

        return type(self)(edge_list[edge_index], edge_weight=self.edge_weight[edge_index], num_node=num_node,
                          view=self.view, meta_dict=meta_dict, **data_dict)

    def subresidue(self, index):
        """
        Return a subgraph based on the specified residues.
        Equivalent to :meth:`residue_mask(index, compact=True) <residue_mask>`.

        Parameters:
            index (array_like): residue index

        Returns:
            Protein

        See also:
            :meth:`Protein.residue_mask`
        """
        return self.residue_mask(index, compact=True)

    @property
    def residue2graph(self):
        """Residue id to graph id mapping."""
        return torch.zeros(self.num_residue, dtype=torch.long, device=self.device)

    @utils.cached_property
    def connected_component_id(self):
        """Connected component id of each residue."""
        node_in, node_out = self.edge_list.t()[:2]
        residue_in, residue_out = self.atom2residue[node_in], self.atom2residue[node_out]
        mask = residue_in != residue_out
        residue_in, residue_out = residue_in[mask], residue_out[mask]
        range = torch.arange(self.num_residue, device=self.device)
        residue_in, residue_out = torch.cat([residue_in, residue_out, range]), \
                                  torch.cat([residue_out, residue_in, range])

        min_neighbor = torch.arange(self.num_residue, device=self.device)
        last = torch.zeros_like(min_neighbor)
        while not torch.equal(min_neighbor, last):
            last = min_neighbor
            min_neighbor = scatter_min(min_neighbor[residue_out], residue_in, dim_size=self.num_residue)[0]
        cc_id = torch.unique(min_neighbor, return_inverse=True)[1]
        return cc_id

    def __repr__(self):
        fields = ["num_atom=%d" % self.num_node, "num_bond=%d" % self.num_edge,
                  "num_residue=%d" % self.num_residue]
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))


class PackedProtein(PackedMolecule, Protein):
    """
    Container for proteins with variadic sizes.
    Support both residue-level and atom-level operations and ensure consistency between two views.

    .. warning::

        Edges of the same graph are guaranteed to be consecutive in the edge list.
        The order of residues must be the same as the protein sequence.
        However, this class doesn't enforce any order on nodes or edges.
        Nodes may have a different order with residues.

    Parameters:
        edge_list (array_like, optional): list of edges of shape :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out, bond_type).
        atom_type (array_like, optional): atom types of shape :math:`(|V|,)`
        bond_type (array_like, optional): bond types of shape :math:`(|E|,)`
        residue_type (array_like, optional): residue types of shape :math:`(|V_{res}|,)`
        view (str, optional): default view for this protein. Can be ``atom`` or ``residue``.
        num_nodes (array_like, optional): number of nodes in each graph
            By default, it will be inferred from the largest id in `edge_list`
        num_edges (array_like, optional): number of edges in each graph
        num_residues (array_like, optional): number of residues in each graph
        offsets (array_like, optional): node id offsets of shape :math:`(|E|,)`.
            If not provided, nodes in `edge_list` should be relative index, i.e., the index in each graph.
            If provided, nodes in `edge_list` should be absolute index, i.e., the index in the packed graph.
    """

    unpacked_type = Protein
    _check_attribute = Protein._check_attribute

    def __init__(self, edge_list=None, atom_type=None, bond_type=None, residue_type=None, view=None, num_nodes=None,
                 num_edges=None, num_residues=None, offsets=None, **kwargs):
        super(PackedProtein, self).__init__(edge_list=edge_list, num_nodes=num_nodes, num_edges=num_edges,
                                            offsets=offsets, atom_type=atom_type, bond_type=bond_type,
                                            residue_type=residue_type, view=view, **kwargs)

        num_residues = torch.as_tensor(num_residues, device=self.device)
        num_cum_residues = num_residues.cumsum(0)

        self.num_residues = num_residues
        self.num_cum_residues = num_cum_residues

    @property
    def num_nodes(self):
        return self.num_atoms

    @num_nodes.setter
    def num_nodes(self, value):
        self.num_atoms = value

    def data_mask(self, node_index=None, edge_index=None, residue_index=None, graph_index=None, include=None,
                  exclude=None):
        data_dict, meta_dict = super(PackedProtein, self).data_mask(node_index, edge_index, graph_index=graph_index,
                                                                    include=include, exclude=exclude)
        residue_mapping = None
        for k, v in data_dict.items():
            for type in meta_dict[k]:
                if type == "residue" and residue_index is not None:
                    if v.is_sparse:
                        v = v.to_dense()[residue_index].to_sparse()
                    else:
                        v = v[residue_index]
                elif type == "residue reference" and residue_index is not None:
                    if residue_mapping is None:
                        residue_mapping = self._get_mapping(residue_index, self.num_residue)
                    v = residue_mapping[v]
            data_dict[k] = v

        return data_dict, meta_dict

    def node_mask(self, index, compact=True):
        index = self._standarize_index(index, self.num_node)
        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
        if compact:
            mapping[index] = torch.arange(len(index), device=self.device)
            num_nodes = self._get_num_xs(index, self.num_cum_nodes)
            offsets = self._get_offsets(num_nodes, self.num_edges)
        else:
            mapping[index] = index
            num_nodes = self.num_nodes
            offsets = self._offsets

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1)
        num_edges = self._get_num_xs(edge_index, self.num_cum_edges)

        if compact:
            data_dict, meta_dict = self.data_mask(index, edge_index)
        else:
            data_dict, meta_dict = self.data_mask(edge_index=edge_index)

        return type(self)(edge_list[edge_index], edge_weight=self.edge_weight[edge_index],
                          num_nodes=num_nodes, num_edges=num_edges, num_residues=self.num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=offsets[edge_index],
                          meta_dict=meta_dict, **data_dict)

    def edge_mask(self, index):
        index = self._standarize_index(index, self.num_edge)
        data_dict, meta_dict = self.data_mask(edge_index=index)
        num_edges = self._get_num_xs(index, self.num_cum_edges)

        return type(self)(self.edge_list[index], edge_weight=self.edge_weight[index],
                          num_nodes=self.num_nodes, num_edges=num_edges, num_residues=self.num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=self._offsets[index],
                          meta_dict=meta_dict, **data_dict)

    def residue_mask(self, index, compact=False):
        """
        Return a masked packed protein based on the specified residues.

        Note the compact option is applied to both residue and atom ids, but not graph ids.

        Parameters:
            index (array_like): residue index
            compact (bool, optional): compact residue ids or not

        Returns:
            PackedProtein
        """
        index = self._standarize_index(index, self.num_residue)
        residue_mapping = -torch.ones(self.num_residue, dtype=torch.long, device=self.device)
        residue_mapping[index] = torch.arange(len(index), device=self.device)

        node_index = residue_mapping[self.atom2residue] >= 0
        node_index = self._standarize_index(node_index, self.num_node)
        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
        if compact:
            mapping[node_index] = torch.arange(len(node_index), device=self.device)
            num_nodes = self._get_num_xs(node_index, self.num_cum_nodes)
            num_residues = self._get_num_xs(index, self.num_cum_residues)
        else:
            mapping[node_index] = node_index
            num_nodes = self.num_nodes
            num_residues = self.num_residues

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1)
        edge_index = self._standarize_index(edge_index, self.num_edge)
        num_edges = self._get_num_xs(edge_index, self.num_cum_edges)
        offsets = self._get_offsets(num_nodes, num_edges)

        if compact:
            data_dict, meta_dict = self.data_mask(node_index, edge_index, residue_index=index)
        else:
            data_dict, meta_dict = self.data_mask(edge_index=edge_index)

        return type(self)(edge_list[edge_index], edge_weight=self.edge_weight[edge_index],
                          num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=offsets,
                          meta_dict=meta_dict, **data_dict)

    def graph_mask(self, index, compact=False):
        index = self._standarize_index(index, self.batch_size)
        graph_mapping = -torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        graph_mapping[index] = torch.arange(len(index), device=self.device)

        node_index = graph_mapping[self.node2graph] >= 0
        node_index = self._standarize_index(node_index, self.num_node)
        residue_index = graph_mapping[self.residue2graph] >= 0
        residue_index = self._standarize_index(residue_index, self.num_residue)
        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
        if compact:
            key = graph_mapping[self.node2graph[node_index]] * self.num_node + node_index
            order = key.argsort()
            node_index = node_index[order]
            key = graph_mapping[self.residue2graph[residue_index]] * self.num_residue + residue_index
            order = key.argsort()
            residue_index = residue_index[order]
            mapping[node_index] = torch.arange(len(node_index), device=self.device)
            num_nodes = self.num_nodes[index]
            num_residues = self.num_residues[index]
        else:
            mapping[node_index] = node_index
            num_nodes = torch.zeros_like(self.num_nodes)
            num_nodes[index] = self.num_nodes[index]
            num_residues = torch.zeros_like(self.num_residues)
            num_residues[index] = self.num_residues[index]

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1)
        edge_index = self._standarize_index(edge_index, self.num_edge)
        if compact:
            key = graph_mapping[self.edge2graph[edge_index]] * self.num_edge + edge_index
            order = key.argsort()
            edge_index = edge_index[order]
            num_edges = self.num_edges[index]
        else:
            num_edges = torch.zeros_like(self.num_edges)
            num_edges[index] = self.num_edges[index]
        offsets = self._get_offsets(num_nodes, num_edges)

        if compact:
            data_dict, meta_dict = self.data_mask(node_index, edge_index,
                                                  residue_index=residue_index, graph_index=index)
        else:
            data_dict, meta_dict = self.data_mask(edge_index=edge_index)

        return type(self)(edge_list[edge_index], edge_weight=self.edge_weight[edge_index],
                          num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=offsets,
                          meta_dict=meta_dict, **data_dict)

    def get_item(self, index):
        node_index = torch.arange(self.num_cum_nodes[index] - self.num_nodes[index], self.num_cum_nodes[index],
                                  device=self.device)
        edge_index = torch.arange(self.num_cum_edges[index] - self.num_edges[index], self.num_cum_edges[index],
                                  device=self.device)
        residue_index = torch.arange(self.num_cum_residues[index] - self.num_residues[index],
                                     self.num_cum_residues[index], device=self.device)
        graph_index = index
        edge_list = self.edge_list[edge_index].clone()
        edge_list[:, :2] -= self._offsets[edge_index].unsqueeze(-1)
        data_dict, meta_dict = self.data_mask(node_index, edge_index,
                                              residue_index=residue_index, graph_index=graph_index)

        return self.unpacked_type(edge_list, edge_weight=self.edge_weight[edge_index], num_node=self.num_nodes[index],
                                  num_relation=self.num_relation, meta_dict=meta_dict, **data_dict)

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_molecule(cls, mols, atom_feature="default", bond_feature="default", residue_feature="default",
                      mol_feature=None, kekulize=False):
        """
        Create a packed protein from a list of RDKit objects.

        Parameters:
            mols (list of rdchem.Mol): molecules
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str or list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        protein = PackedMolecule.from_molecule(mols, atom_feature=atom_feature, bond_feature=bond_feature,
                                               mol_feature=mol_feature, with_hydrogen=False, kekulize=kekulize)
        residue_feature = cls._standarize_option(residue_feature)

        residue_type = []
        atom_name = []
        is_hetero_atom = []
        occupancy = []
        b_factor = []
        atom2residue = []
        residue_number = []
        insertion_code = []
        chain_id = []
        _residue_feature = []
        last_residue = None
        num_residues = []
        num_cum_residue = 0

        mols = mols + [cls.dummy_protein]
        for mol in mols:
            if mol is None:
                mol = cls.empty_mol

            if kekulize:
                Chem.Kekulize(mol)

            for atom in mol.GetAtoms():
                residue = atom.GetPDBResidueInfo()
                number = residue.GetResidueNumber()
                code = residue.GetInsertionCode()
                type = residue.GetResidueName().strip()
                canonical_residue = (number, code, type)
                if canonical_residue != last_residue:
                    last_residue = canonical_residue
                    if type not in cls.residue2id:
                        warnings.warn("Unknown residue `%s`. Treat as glycine" % type)
                        type = "GLY"
                    residue_type.append(cls.residue2id[type])
                    residue_number.append(number)
                    insertion_code.append(cls.alphabet2id[residue.GetInsertionCode()])
                    chain_id.append(cls.alphabet2id[residue.GetChainId()])
                    feature = []
                    for name in residue_feature:
                        func = R.get("features.residue.%s" % name)
                        feature += func(residue)
                    _residue_feature.append(feature)
                name = residue.GetName().strip()
                if name not in cls.atom_name2id:
                    name = "UNK"
                atom_name.append(cls.atom_name2id[name])
                is_hetero_atom.append(residue.GetIsHeteroAtom())
                occupancy.append(residue.GetOccupancy())
                b_factor.append(residue.GetTempFactor())
                atom2residue.append(len(residue_type) - 1)

            num_residues.append(len(residue_type) - num_cum_residue)
            num_cum_residue = len(residue_type)

        residue_type = torch.tensor(residue_type)[:-1]
        atom_name = torch.tensor(atom_name)[:-5]
        is_hetero_atom = torch.tensor(is_hetero_atom)[:-5]
        occupancy = torch.tensor(occupancy)[:-5]
        b_factor = torch.tensor(b_factor)[:-5]
        atom2residue = torch.tensor(atom2residue)[:-5]
        residue_number = torch.tensor(residue_number)[:-1]
        insertion_code = torch.tensor(insertion_code)[:-1]
        chain_id = torch.tensor(chain_id)[:-1]
        if len(residue_feature) > 0:
            _residue_feature = torch.tensor(_residue_feature)[:-1]
        else:
            _residue_feature = None

        num_residues = num_residues[:-1]

        return cls(protein.edge_list, residue_type=residue_type,
                   num_nodes=protein.num_nodes, num_edges=protein.num_edges, num_residues=num_residues,
                   atom_name=atom_name, atom2residue=atom2residue, residue_feature=_residue_feature,
                   is_hetero_atom=is_hetero_atom, occupancy=occupancy, b_factor=b_factor,
                   residue_number=residue_number, insertion_code=insertion_code, chain_id=chain_id,
                   offsets=protein._offsets, meta_dict=protein.meta_dict, **protein.data_dict)

    @classmethod
    def _residue_from_sequence(cls, sequences):
        num_residues = []
        residue_type = []
        residue_feature = []
        sequences = sequences + ["G"]
        for sequence in sequences:
            for residue in sequence:
                if residue not in cls.residue_symbol2id:
                    warnings.warn("Unknown residue symbol `%s`. Treat as glycine" % residue)
                    residue = "G"
                residue_type.append(cls.residue_symbol2id[residue])
                residue_feature.append(feature.onehot(residue, cls.residue_symbol2id, allow_unknown=True))
            num_residues.append(len(sequence))

        residue_type = residue_type[:-1]
        residue_feature = torch.tensor(residue_feature)[:-1]

        edge_list = torch.zeros(0, 3, dtype=torch.long)
        num_nodes = [0] * (len(sequences) - 1)
        num_edges = [0] * (len(sequences) - 1)
        num_residues = num_residues[:-1]

        return cls(edge_list=edge_list, atom_type=[], bond_type=[], residue_type=residue_type,
                   num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues,
                   residue_feature=residue_feature)

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_sequence(cls, sequences, atom_feature="default", bond_feature="default", residue_feature="default",
                      mol_feature=None, kekulize=False):
        """
        Create a packed protein from a list of sequences.

        .. note::

            It takes considerable time to construct proteins with a large number of atoms and bonds.
            If you only need residue information, you may speed up the construction by setting
            ``atom_feature`` and ``bond_feature`` to ``None``.

        Parameters:
            sequences (str): list of protein sequences
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str or list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if atom_feature is None and bond_feature is None and residue_feature == "default":
            return cls._residue_from_sequence(sequences)

        mols = []
        for sequence in sequences:
            mol = Chem.MolFromSequence(sequence)
            if mol is None:
                raise ValueError("Invalid sequence `%s`" % sequence)
            mols.append(mol)

        return cls.from_molecule(mols, atom_feature, bond_feature, residue_feature, mol_feature, kekulize)

    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_pdb(cls, pdb_files, atom_feature="default", bond_feature="default", residue_feature="default",
                 mol_feature=None, kekulize=False):
        """
        Create a protein from a list of PDB files.

        Parameters:
            pdb_files (str): list of file names
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str, list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        mols = []
        for pdb_file in pdb_files:
            mol = Chem.MolFromPDBFile(pdb_file)
            mols.append(mol)

        return cls.from_molecule(mols, atom_feature, bond_feature, residue_feature, mol_feature, kekulize)

    def to_molecule(self, ignore_error=False):
        mols = super(PackedProtein, self).to_molecule(ignore_error)

        residue_type = self.residue_type.tolist()
        atom_name = self.atom_name.tolist()
        atom2residue = self.atom2residue.tolist()
        is_hetero_atom = self.is_hetero_atom.tolist()
        occupancy = self.occupancy.tolist()
        b_factor = self.b_factor.tolist()
        residue_number = self.residue_number.tolist()
        chain_id = self.chain_id.tolist()
        insertion_code = self.insertion_code.tolist()
        num_cum_nodes = [0] + self.num_cum_nodes.tolist()

        for i, mol in enumerate(mols):
            for j, atom in enumerate(mol.GetAtoms(), num_cum_nodes[i]):
                r = atom2residue[j]
                residue = Chem.AtomPDBResidueInfo()
                residue.SetResidueNumber(residue_number[r])
                residue.SetChainId(self.id2alphabet[chain_id[r]])
                residue.SetInsertionCode(self.id2alphabet[insertion_code[r]])
                residue.SetName(" %-3s" % self.id2atom_name[atom_name[j]])
                residue.SetResidueName(self.id2residue[residue_type[r]])
                residue.SetIsHeteroAtom(is_hetero_atom[j])
                residue.SetOccupancy(occupancy[j])
                residue.SetTempFactor(b_factor[j])
                atom.SetPDBResidueInfo(residue)

        return mols

    def to_sequence(self):
        """
        Return a list of sequences.

        Returns:
            list of str
        """
        residue_type = self.residue_type.tolist()
        cc_id = self.connected_component_id.tolist()
        num_cum_residues = [0] + self.num_cum_residues.tolist()
        sequences = []
        for i in range(self.batch_size):
            sequence = []
            for j in range(num_cum_residues[i], num_cum_residues[i + 1]):
                if j > num_cum_residues[i] and cc_id[j] > cc_id[j - 1]:
                    sequence.append(".")
                sequence.append(self.id2residue_symbol[residue_type[j]])
            sequence = "".join(sequence)
            sequences.append(sequence)
        return sequences

    def to_pdb(self, pdb_files):
        """
        Write this packed protein to several pdb files.

        Parameters:
            pdb_files (list of str): list of file names
        """
        mols = self.to_molecule()
        for mol, pdb_file in zip(mols, pdb_files):
            Chem.MolToPDBFile(mol, pdb_file, flavor=10)

    def merge(self, graph2graph):
        graph2graph = torch.as_tensor(graph2graph, dtype=torch.long, device=self.device)
        # coalesce arbitrary graph IDs to [0, n)
        _, graph2graph = torch.unique(graph2graph, return_inverse=True)

        graph_key = graph2graph * self.batch_size + torch.arange(self.batch_size, device=self.device)
        graph_index = graph_key.argsort()
        graph = self.subbatch(graph_index)
        graph2graph = graph2graph[graph_index]

        num_graph = graph2graph[-1] + 1
        num_nodes = scatter_add(graph.num_nodes, graph2graph, dim_size=num_graph)
        num_edges = scatter_add(graph.num_edges, graph2graph, dim_size=num_graph)
        num_residues = scatter_add(graph.num_residues, graph2graph, dim_size=num_graph)
        offsets = self._get_offsets(num_nodes, num_edges)

        data_dict, meta_dict = graph.data_mask(exclude="graph")

        return type(self)(graph.edge_list, edge_weight=graph.edge_weight, num_nodes=num_nodes,
                          num_edges=num_edges, num_residues=num_residues, view=self.view, offsets=offsets,
                          meta_dict=meta_dict, **data_dict)

    def repeat(self, count):
        num_nodes = self.num_nodes.repeat(count)
        num_edges = self.num_edges.repeat(count)
        num_residues = self.num_residues.repeat(count)
        offsets = self._get_offsets(num_nodes, num_edges)
        edge_list = self.edge_list.repeat(count, 1)
        edge_list[:, :2] += (offsets - self._offsets.repeat(count)).unsqueeze(-1)

        data_dict = {}
        for k, v in self.data_dict.items():
            shape = [1] * v.ndim
            shape[0] = count
            length = len(v)
            v = v.repeat(shape)
            for _type in self.meta_dict[k]:
                if _type == "node reference":
                    pack_offsets = torch.arange(count, device=self.device) * self.num_node
                    v = v + pack_offsets.repeat_interleave(length)
                elif _type == "edge reference":
                    pack_offsets = torch.arange(count, device=self.device) * self.num_edge
                    v = v + pack_offsets.repeat_interleave(length)
                elif _type == "residue reference":
                    pack_offsets = torch.arange(count, device=self.device) * self.num_residue
                    v = v + pack_offsets.repeat_interleave(length)
                elif _type == "graph reference":
                    pack_offsets = torch.arange(count, device=self.device) * self.batch_size
                    v = v + pack_offsets.repeat_interleave(length)
            data_dict[k] = v

        return type(self)(edge_list, edge_weight=self.edge_weight.repeat(count),
                          num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues, view=self.view,
                          num_relation=self.num_relation, offsets=offsets,
                          meta_dict=self.meta_dict, **data_dict)

    def repeat_interleave(self, repeats):
        repeats = torch.as_tensor(repeats, dtype=torch.long, device=self.device)
        if repeats.numel() == 1:
            repeats = repeats * torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        num_nodes = self.num_nodes.repeat_interleave(repeats)
        num_edges = self.num_edges.repeat_interleave(repeats)
        num_residues = self.num_residues.repeat_interleave(repeats)
        num_cum_nodes = num_nodes.cumsum(0)
        num_cum_edges = num_edges.cumsum(0)
        num_cum_residues = num_residues.cumsum(0)
        num_node = num_nodes.sum()
        num_edge = num_edges.sum()
        num_residue = num_residues.sum()
        batch_size = repeats.sum()
        num_graphs = torch.ones(batch_size, device=self.device)

        # special case 1: graphs[i] may have no node or no edge
        # special case 2: repeats[i] may be 0
        cum_repeats_shifted = repeats.cumsum(0) - repeats
        graph_mask = cum_repeats_shifted < batch_size
        cum_repeats_shifted = cum_repeats_shifted[graph_mask]

        index = num_cum_nodes - num_nodes
        index = torch.cat([index, index[cum_repeats_shifted]])
        value = torch.cat([-num_nodes, self.num_nodes[graph_mask]])
        mask = index < num_node
        node_index = scatter_add(value[mask], index[mask], dim_size=num_node)
        node_index = (node_index + 1).cumsum(0) - 1

        index = num_cum_edges - num_edges
        index = torch.cat([index, index[cum_repeats_shifted]])
        value = torch.cat([-num_edges, self.num_edges[graph_mask]])
        mask = index < num_edge
        edge_index = scatter_add(value[mask], index[mask], dim_size=num_edge)
        edge_index = (edge_index + 1).cumsum(0) - 1

        index = num_cum_residues - num_residues
        index = torch.cat([index, index[cum_repeats_shifted]])
        value = torch.cat([-num_residues, self.num_residues[graph_mask]])
        mask = index < num_residue
        residue_index = scatter_add(value[mask], index[mask], dim_size=num_residue)
        residue_index = (residue_index + 1).cumsum(0) - 1

        graph_index = torch.repeat_interleave(repeats)

        offsets = self._get_offsets(num_nodes, num_edges)
        edge_list = self.edge_list[edge_index]
        edge_list[:, :2] += (offsets - self._offsets[edge_index]).unsqueeze(-1)

        node_offsets = None
        edge_offsets = None
        residue_offsets = None
        graph_offsets = None
        data_dict = {}
        for k, v in self.data_dict.items():
            num_xs = None
            pack_offsets = None
            for _type in self.meta_dict[k]:
                if _type == "node":
                    v = v[node_index]
                    num_xs = num_nodes
                elif _type == "edge":
                    v = v[edge_index]
                    num_xs = num_edges
                elif _type == "residue":
                    v = v[residue_index]
                    num_xs = num_residues
                elif _type == "graph":
                    v = v[graph_index]
                    num_xs = num_graphs
                elif _type == "node reference":
                    if node_offsets is None:
                        node_offsets = self._get_repeat_pack_offsets(self.num_nodes, repeats)
                    pack_offsets = node_offsets
                elif _type == "edge reference":
                    if edge_offsets is None:
                        edge_offsets = self._get_repeat_pack_offsets(self.num_edges, repeats)
                    pack_offsets = edge_offsets
                elif _type == "residue reference":
                    if residue_offsets is None:
                        residue_offsets = self._get_repeat_pack_offsets(self.num_residues, repeats)
                    pack_offsets = residue_offsets
                elif _type == "graph reference":
                    if graph_offsets is None:
                        graph_offsets = self._get_repeat_pack_offsets(num_graphs, repeats)
                    pack_offsets = graph_offsets
            # add offsets to make references point to indexes in their own graph
            if num_xs is not None and pack_offsets is not None:
                v = v + pack_offsets.repeat_interleave(num_xs)
            data_dict[k] = v

        return type(self)(edge_list, edge_weight=self.edge_weight[edge_index],
                          num_nodes=num_nodes, num_edges=num_edges, num_residues=num_residues, view=self.view,
                          num_relation=self.num_relation, offsets=offsets, meta_dict=self.meta_dict, **data_dict)

    def undirected(self, add_inverse=True):
        undirected = PackedMolecule.undirected(self, add_inverse=add_inverse)

        return type(self)(undirected.edge_list, edge_weight=undirected.edge_weight,
                          num_nodes=undirected.num_nodes, num_edges=undirected.num_edges,
                          num_residues=self.num_residues, view=self.view, num_relation=undirected.num_relation,
                          offsets=undirected._offsets, meta_dict=undirected.meta_dict, **undirected.data_dict)

    def detach(self):
        return type(self)(self.edge_list.detach(), edge_weight=self.edge_weight.detach(),
                          num_nodes=self.num_nodes, num_edges=self.num_edges, num_residues=self.num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=self._offsets,
                          meta_dict=self.meta_dict, **utils.detach(self.data_dict))

    def clone(self):
        return type(self)(self.edge_list.clone(), edge_weight=self.edge_weight.clone(),
                          num_nodes=self.num_nodes, num_edges=self.num_edges, num_residues=self.num_residues,
                          view=self.view, num_relation=self.num_relation, offsets=self._offsets,
                          meta_dict=self.meta_dict, **utils.clone(self.data_dict))

    def cuda(self, *args, **kwargs):
        edge_list = self.edge_list.cuda(*args, **kwargs)

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight,
                              num_nodes=self.num_nodes, num_edges=self.num_edges, num_residues=self.num_residues,
                              view=self.view, num_relation=self.num_relation, offsets=self._offsets,
                              meta_dict=self.meta_dict, **utils.cuda(self.data_dict, *args, **kwargs))

    def cpu(self):
        edge_list = self.edge_list.cpu()

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight,
                              num_nodes=self.num_nodes, num_edges=self.num_edges, num_residues=self.num_residues,
                              view=self.view, num_relation=self.num_relation, offsets=self._offsets,
                              meta_dict=self.meta_dict, **utils.cpu(self.data_dict))

    @utils.cached_property
    def residue2graph(self):
        """Residue id to graph id mapping."""
        range = torch.arange(self.batch_size, device=self.device)
        residue2graph = range.repeat_interleave(self.num_residues)
        return residue2graph

    @utils.cached_property
    def connected_component_id(self):
        cc_id = super(PackedProtein, self).connected_component_id
        cc_id_offsets = scatter_min(cc_id, self.residue2graph, dim_size=self.num_residue)[0][self.residue2graph]
        cc_id = cc_id - cc_id_offsets
        return cc_id

    def __repr__(self):
        fields = ["batch_size=%d" % self.batch_size,
                  "num_atoms=%s" % pretty.long_array(self.num_nodes.tolist()),
                  "num_bonds=%s" % pretty.long_array(self.num_edges.tolist()),
                  "num_residues=%s" % pretty.long_array(self.num_residues.tolist())]
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))


Protein.packed_type = PackedProtein
