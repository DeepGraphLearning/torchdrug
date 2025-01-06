import io
import os
import json
import jinja2
from PIL import Image

from rdkit.Chem import AllChem, Draw


path = os.path.join(os.path.dirname(__file__), "template")


def reaction(reactants, products, save_file=None, figure_size=(3, 3), atom_map=False):
    """
    Visualize a chemical reaction.

    Parameters:
        reactants (list of Molecule): list of reactants
        products (list of Molecule): list of products
        save_file (str, optional): save_file (str, optional): ``png`` file to save visualization.
                If not provided, show the figure in window.
        figure_size (tuple of int, optional): width and height of the figure
        atom_map (bool, optional): visualize atom mapping or not
    """
    rxn = AllChem.ChemicalReaction()
    for reactant in reactants:
        mol = reactant.to_molecule()
        if not atom_map:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        rxn.AddReactantTemplate(mol)
    for product in products:
        mol = product.to_molecule()
        if not atom_map:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        rxn.AddProductTemplate(mol)
    size = [100 * s for s in figure_size]
    img = Draw.ReactionToImage(rxn, size)

    if save_file is None:
        img.show()
    else:
        img.save(save_file)


def highlight(molecule, atoms=None, bonds=None, atom_colors=None, bond_colors=None, save_file=None, figure_size=(3, 3),
              atom_map=False):
    """
    Visualize a molecule with highlighted atoms or bonds.

    Parameters:
        molecule (Molecule): molecule to visualize
        atoms (list of int): indexes of atoms to highlight
        bonds (list of int): indexes of bonds to highlight
        atom_colors (tuple or dict): highlight color for atoms.
            Can be a tuple of 3 float between 0 and 1, or a dict that maps each index to a different color.
        bond_colors (tuple or dict): highlight color for bonds.
            Can be a tuple of 3 float between 0 and 1, or a dict that maps each index to a different color.
        save_file (str, optional): save_file (str, optional): ``png`` file to save visualization.
                If not provided, show the figure in window.
        figure_size (tuple of int, optional): width and height of the figure
        atom_map (bool, optional): visualize atom mapping or not
    """
    if not isinstance(atom_colors, dict):
        atom_colors = dict.fromkeys(atoms, atom_colors)
    if not isinstance(bond_colors, dict):
        bond_colors = dict.fromkeys(bonds, bond_colors)

    mol = molecule.to_molecule()
    if not atom_map:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    size = [100 * s for s in figure_size]
    canvas = Draw.rdMolDraw2D.MolDraw2DCairo(*size)
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(canvas, mol, highlightAtoms=atoms, highlightBonds=bonds,
                                            highlightAtomColors=atom_colors, highlightBondColors=bond_colors)

    if save_file is None:
        stream = io.BytesIO(canvas.GetDrawingText())
        img = Image.open(stream)
        img.show()
    else:
        canvas.WriteDrawingText(save_file)


def echarts(graph, title=None, node_colors=None, edge_colors=None, node_labels=None, relation_labels=None,
            node_types=None, type_labels=None, dynamic_size=False, dynamic_width=False, save_file=None):
    """
    Visualize a graph in ECharts.

    Parameters:
        graph (Graph): graph to visualize
        title (str, optional): title of the graph
        node_colors (dict, optional): specify colors for some nodes.
            Each color is either a tuple of 3 integers between 0 and 255, or a hex color code.
        edge_colors (dict, optional): specify colors for some edges.
            Each color is either a tuple of 3 integers between 0 and 255, or a hex color code.
        node_labels (list of str, optional): labels for each node
        relation_labels (list of str, optional): labels for each relation
        node_types (list of int, optional): type for each node
        type_labels (list of str, optional): labels for each node type
        dynamic_size (bool, optional): if true, set the size of nodes based on the logarithm of degrees
        dynamic_width (bool, optional): if true, set the width of edges based on the edge weights
        save_file (str, optional): ``html`` file to save visualization, accompanied by a ``json`` file
    """
    if dynamic_size:
        symbol_size = (graph.degree_in + graph.degree_out + 2).log()
        symbol_size = symbol_size / symbol_size.mean() * 10
        symbol_size = symbol_size.tolist()
    else:
        symbol_size = [10] * graph.num_node
    nodes = []
    node_colors = node_colors or {}
    for i in range(graph.num_node):
        node = {
            "id": i,
            "symbolSize": symbol_size[i],
        }
        if i in node_colors:
            color = node_colors[i]
            if isinstance(color, tuple):
                color = "rgb%s" % (color,)
            node["itemStyle"] = {"color": color}
        if node_labels:
            node["name"] = node_labels[i]
        if node_types:
            node["category"] = node_types[i]
        nodes.append(node)

    if dynamic_width:
        width = graph.edge_weight / graph.edge_weight.mean() * 3
        width = width.tolist()
    else:
        width = [3] * graph.num_edge
    edges = []
    if graph.num_relation:
        node_in, node_out, relation = graph.edge_list.t().tolist()
    else:
        node_in, node_out = graph.edge_list.t().tolist()
        relation = None
    edge_colors = edge_colors or {}
    for i in range(graph.num_edge):
        edge = {
            "source": node_in[i],
            "target": node_out[i],
            "lineStyle": {"width": width[i]},
        }
        if i in edge_colors:
            color = edge_colors[i]
            if isinstance(color, tuple):
                color = "rgb%s" % (color,)
            edge["lineStyle"] = {"color": color}
        if relation_labels:
            edge["value"] = relation_labels[relation[i]]
        edges.append(edge)

    json_file = os.path.splitext(save_file)[0] + ".json"
    data = {
        "title": title,
        "nodes": nodes,
        "edges": edges,
    }
    if type_labels:
        data["categories"] = [{"name": label} for label in type_labels]
    variables = {
        "data_file": os.path.basename(json_file),
        "show_label": "true" if node_labels else "false",
    }
    with open(os.path.join(path, "echarts.html"), "r") as fin, open(save_file, "w") as fout:
        template = jinja2.Template(fin.read())
        instance = template.render(variables)
        fout.write(instance)
    with open(json_file, "w") as fout:
        json.dump(data, fout, sort_keys=True, indent=4)