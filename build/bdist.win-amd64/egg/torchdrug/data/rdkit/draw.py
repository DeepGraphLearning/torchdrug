from matplotlib import pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.Chem.Draw import mplCanvas


class Canvas(mplCanvas.Canvas):

    def __init__(self, ax, name="", imageType="png"):
        self._name = name
        if ax is None:
            size = (3, 3)
            self._figure = plt.figure(figsize=size)
            self._axes = self._figure.add_axes([0, 0, 1, 1])
        else:
            bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
            size = (bbox.width, bbox.height)
            self._figure = ax.figure
            self._axes = ax
        self._axes.set_axis_off()
        # these are rdkit internal size and dpi
        self.size = tuple(s * 100 for s in size)
        self._dpi = max(self.size)


def MolToMPL(mol, ax=None, kekulize=True, wedgeBonds=True, imageType=None, fitImage=False,
             options=None, **kwargs):
    """Generates a drawing of a molecule on a matplotlib canvas."""
    if not mol:
        raise ValueError("Null molecule provided")

    canvas = Canvas(ax)
    if options is None:
        options = DrawingOptions()
        options.bgColor = None
    if fitImage:
        options.dotsPerAngstrom = int(min(canvas.size) / 10)
    options.wedgeDashedBonds = wedgeBonds
    drawer = MolDrawing(canvas=canvas, drawingOptions=options)
    omol = mol
    if kekulize:
        mol = Chem.Mol(mol.ToBinary())
        Chem.Kekulize(mol)

    if not mol.GetNumConformers():
        AllChem.Compute2DCoords(mol)

    drawer.AddMol(mol, **kwargs)
    omol._atomPs = drawer.atomPs[mol]
    for k, v in omol._atomPs.items():
        omol._atomPs[k] = canvas.rescalePt(v)
    return canvas._figure