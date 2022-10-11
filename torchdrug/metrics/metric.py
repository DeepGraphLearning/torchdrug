import torch
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors

from torchdrug import utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.metrics.rdkit import sascorer


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


@R.register("metrics.plogp")
def penalized_logP(pred):
    """
    Logarithm of partition coefficient, penalized by cycle length and synthetic accessibility.

    Parameters:
        pred (PackedMolecule): molecules to evaluate
    """
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


@R.register("metrics.variadic_auroc")
def variadic_area_under_roc(pred, target, size):
    """
    Area under receiver operating characteristic curve (ROC) for sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        pred (Tensor): prediction of shape :math:`(B,)`
        target (Tensor): target of shape :math:`(B,)`.
        size (Tensor): size of sets of shape :math:`(N,)`
    """
    index2graph = functional._size_to_index(size)
    _, order = functional.variadic_sort(pred, size, descending=True)
    cum_size = (size.cumsum(0) - size)[index2graph]
    target = target[order + cum_size]
    total_hit = functional.variadic_sum(target, size)
    total_hit = total_hit.cumsum(0) - total_hit
    hit = target.cumsum(0) - total_hit[index2graph]
    hit = torch.where(target == 0, hit, torch.zeros_like(hit))
    all = functional.variadic_sum((target == 0).float(), size) * \
            functional.variadic_sum((target == 1).float(), size)
    auroc = functional.variadic_sum(hit, size) / (all + 1e-10)
    return auroc


@R.register("metrics.variadic_auprc")
def variadic_area_under_prc(pred, target, size):
    """
    Area under precision-recall curve (PRC) for sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        pred (Tensor): prediction of shape :math:`(B,)`
        target (Tensor): target of shape :math:`(B,)`.
        size (Tensor): size of sets of shape :math:`(N,)`
    """
    index2graph = functional._size_to_index(size)
    _, order = functional.variadic_sort(pred, size, descending=True)
    cum_size = (size.cumsum(0) - size)[index2graph]
    target = target[order + cum_size]
    total_hit = functional.variadic_sum(target, size)
    total_hit = total_hit.cumsum(0) - total_hit
    hit = target.cumsum(0) - total_hit[index2graph]
    total = torch.ones_like(target).cumsum(0) - (size.cumsum(0) - size)[index2graph]
    precision = hit / total
    precision = torch.where(target == 1, precision, torch.zeros_like(precision))
    auprc = functional.variadic_sum(precision, size) / \
            (functional.variadic_sum((target == 1).float(), size) + 1e-10)
    return auprc
    

@R.register("metrics.f1_max")
def f1_max(pred, target):
    """
    F1 score with the optimal threshold.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()
    

@R.register("metrics.accuracy")
def accuracy(pred, target):
    """
    Classification accuracy.

    Suppose there are :math:`N` sets and :math:`C` categories.

    Parameters:
        pred (Tensor): prediction of shape :math:`(N, C)`
        target (Tensor): target of shape :math:`(N,)`
    """
    return (pred.argmax(dim=-1) == target).float().mean()


@R.register("metrics.variadic_accuracy")
def variadic_accuracy(input, target, size):
    """
    Classification accuracy for categories with variadic sizes.

    Suppose there are :math:`N` samples, and the number of categories in all samples is summed to :math:`B`.

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


@R.register("metrics.variadic_top_precision")
def variadic_top_precision(pred, target, size, k):
    """
    Top-k precision for sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        pred (Tensor): prediction of shape :math:`(B,)`
        target (Tensor): target of shape :math:`(B,)`
        size (Tensor): size of sets of shape :math:`(N,)`
        k (LongTensor): the k in "top-k" for different sets of shape :math:`(N,)`
    """
    index = functional.variadic_topk(pred, size, k, largest=True)[1]
    index = index + (size.cumsum(0) - size).repeat_interleave(k)
    precision = functional.variadic_sum(target[index], k) / k
    precision[size < k] = 0
    return precision


@R.register("metrics.mcc")
def matthews_corrcoef(pred, target):
    """
    Matthews correlation coefficient between prediction and target.

    Definition follows matthews_corrcoef for K classes in sklearn.
    For details, see: `https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef`

    Parameters:
        pred (Tensor): prediction of shape :math: `(N, K)`
        target (Tensor): target of shape :math: `(N,)`
    """
    num_class = pred.size(-1)
    pred = pred.argmax(-1)
    ones = torch.ones(len(target), device=pred.device)
    confusion_matrix = scatter_add(ones, target * num_class + pred, dim=0, dim_size=num_class ** 2)
    confusion_matrix = confusion_matrix.view(num_class, num_class)
    t = confusion_matrix.sum(dim=1)
    p = confusion_matrix.sum(dim=0)
    c = confusion_matrix.trace()
    s = confusion_matrix.sum()
    return (c * s - t @ p) / ((s * s - p @ p) * (s * s - t @ t) + 1e-10).sqrt()


@R.register("metrics.pearsonr")
def pearsonr(pred, target):
    """
    Pearson correlation between prediction and target.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """
    pred_mean = pred.float().mean()
    target_mean = target.float().mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    pred_normalized = pred_centered / pred_centered.norm(2)
    target_normalized = target_centered / target_centered.norm(2)
    pearsonr = pred_normalized @ target_normalized
    return pearsonr


@R.register("metrics.spearmanr")
def spearmanr(pred, target):
    """
    Spearman correlation between prediction and target.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """

    def get_ranking(input):
        input_set, input_inverse = input.unique(return_inverse=True)
        order = input_inverse.argsort()
        ranking = torch.zeros(len(input_inverse), device=input.device)
        ranking[order] = torch.arange(1, len(input) + 1, dtype=torch.float, device=input.device)

        # for elements that have the same value, replace their rankings with the mean of their rankings
        mean_ranking = scatter_mean(ranking, input_inverse, dim=0, dim_size=len(input_set))
        ranking = mean_ranking[input_inverse]
        return ranking

    pred = get_ranking(pred)
    target = get_ranking(target)
    covariance = (pred * target).mean() - pred.mean() * target.mean()
    pred_std = pred.std(unbiased=False)
    target_std = target.std(unbiased=False)
    spearmanr = covariance / (pred_std * target_std + 1e-10)
    return spearmanr
