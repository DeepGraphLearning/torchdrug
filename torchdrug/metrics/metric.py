import os
import sys

import torch
from torch.nn import functional as F
from torch_scatter import scatter_max
import networkx as nx
from rdkit import Chem
from rdkit.Chem import RDConfig, Descriptors

from torchdrug import utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("metrics.auroc")
def area_under_roc(pred, target):
    """
    Area under receiver operating characteristic curve (ROC).

    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    order = pred.argsort(descending=True)
    target = target[order]
    hit = target.cumsum(0)
    all = (target == 0).sum() * (target == 1).sum()
    auroc = hit[target == 0].sum() / (all + 1e-10)
    return auroc


@R.register("metrics.auprc")
def area_under_prc(pred, target):
    """
    Area under precision-recall curve (PRC).

    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    order = pred.argsort(descending=True)
    target = target[order]
    precision = target.cumsum(0) / torch.arange(1, len(target) + 1, device=target.device)
    auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
    return auprc


@R.register("metrics.r2")
def r2(pred, target):
    """
    :math:`R^2` regression score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): targets of shape :math:`(n,)`
    """
    total = torch.var(target, unbiased=False)
    residual = F.mse_loss(pred, target)
    return 1 - residual / total


@R.register("metrics.logp")
def logP(pred):
    """
    Logarithm of partition coefficient between octanol and water for a compound.

    Parameters:
        pred (PackedMolecule): molecules to evaluate
    """
    logp = []
    for mol in pred:
        mol = mol.to_molecule()
        try:
            with utils.no_rdkit_log():
                mol.UpdatePropertyCache()
                score = Descriptors.MolLogP(mol)
        except Chem.AtomValenceException:
            score = 0
        logp.append(score)

    return torch.tensor(logp, dtype=torch.float, device=pred.device)


@R.register("metrics.penalized_logp")
def penalized_logP(pred):
    """
    Logarithm of partition coefficient, penalized by cycle length and synthetic accessibility.

    Parameters:
        pred (PackedMolecule): molecules to evaluate
    """
    if "sascorer" not in sys.modules:
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer

    # statistics from ZINC250k
    logp_mean = 2.4570953396190123
    logp_std = 1.434324401111988
    sa_mean = 3.0525811293166134
    sa_std = 0.8335207024513095
    cycle_mean = 0.0485696876403053
    cycle_std = 0.2860212110245455

    plogp = []
    for mol in pred:
        cycles = nx.cycle_basis(nx.Graph(mol.edge_list[:, :2].tolist()))
        if cycles:
            max_cycle = max([len(cycle) for cycle in cycles])
            cycle = max(0, max_cycle - 6)
        else:
            cycle = 0
        mol = mol.to_molecule()
        try:
            with utils.no_rdkit_log():
                mol.UpdatePropertyCache()
                Chem.GetSymmSSSR(mol)
                logp = Descriptors.MolLogP(mol)
                sa = sascorer.calculateScore(mol)
            logp = (logp - logp_mean) / logp_std
            sa = (sa - sa_mean) / sa_std
            cycle = (cycle - cycle_mean) / cycle_std
            score = logp - sa - cycle
        except Chem.AtomValenceException:
            score = -30
        plogp.append(score)

    return torch.tensor(plogp, dtype=torch.float, device=pred.device)


@R.register("metrics.SA")
def SA(pred):
    """
    Synthetic accesibility score.

    Parameters:
        pred (PackedMolecule): molecules to evaluate
    """
    sa = []
    for mol in pred:
        with utils.no_rdkit_log():
            score = sascorer.calculateScore(mol.to_molecule())
        sa.append(score)

    return torch.tensor(sa, dtype=torch.float, device=pred.device)


@R.register("metrics.qed")
def QED(pred):
    """
    Quantitative estimation of drug-likeness.

    Parameters:
        pred (PackedMolecule): molecules to evaluate
    """
    qed = []
    for mol in pred:
        try:
            with utils.no_rdkit_log():
                score = Descriptors.qed(mol.to_molecule())
        except Chem.AtomValenceException:
            score = -1
        qed.append(score)

    return torch.tensor(qed, dtype=torch.float, device=pred.device)


@R.register("metrics.validity")
def chemical_validity(pred):
    """
    Chemical validity of molecules.

    Parameters:
        pred (PackedMolecule): molecules to evaluate
    """
    validity = []
    for i, mol in enumerate(pred):
        with utils.no_rdkit_log():
            smiles = mol.to_smiles()
            mol = Chem.MolFromSmiles(smiles)
        validity.append(1 if mol else 0)

    return torch.tensor(validity, dtype=torch.float, device=pred.device)


def variadic_accuracy(input, target, size):
    """
    Compute classification accuracy over variadic sizes of categories.

    Suppose there are :math:`N` samples, and the number of categories in all samples is summed to :math`B`.

    Parameters:
        input (Tensor): prediction of shape :math:`(B,)`
        target (Tensor): target of shape :math:`(N,)`. Each target is a relative index in a sample.
        size (Tensor): number of categories of shape :math:`(N,)`
    """
    index2graph = functional._size_to_index(size)

    input_class = scatter_max(input, index2graph)[1]
    target_index = target + size.cumsum(0) - size
    accuracy = (input_class == target_index).float()
    return accuracy