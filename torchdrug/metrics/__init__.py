from .metric import area_under_roc, area_under_prc, r2, QED, logP, penalized_logP, SA, chemical_validity, \
    variadic_accuracy

# alias
AUROC = area_under_roc
AUPRC = area_under_prc

__all__ = [
    "area_under_roc", "area_under_prc", "r2", "QED", "logP", "penalized_logP", "SA", "chemical_validity",
    "variadic_accuracy",
    "AUROC", "AUPRC",
]