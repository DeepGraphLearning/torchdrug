from .bace import BACE
from .bbbp import BBBP
from .cep import CEP
from .clintox import ClinTox
from .delaney import Delaney
from .freesolv import FreeSolv
from .hiv import HIV
from .lipophilicity import Lipophilicity
from .malaria import Malaria
from .moses import MOSES
from .muv import MUV
from .opv import OPV
from .qm8 import QM8
from .qm9 import QM9
from .sider import SIDER
from .tox21 import Tox21
from .toxcast import ToxCast
from .uspto50k import USPTO50k
from .zinc250k import ZINC250k
from .zinc2m import ZINC2m
from .pubchem110m import PubChem110m

from .chembl_filtered import ChEMBLFiltered

from .fb15k import FB15k, FB15k237
from .wn18 import WN18, WN18RR
from .hetionet import Hetionet

from .cora import Cora
from .citeseer import CiteSeer
from .pubmed import PubMed

__all__ = [
    "BACE", "BBBP", "CEP", "ChEMBLFiltered", "ClinTox", "Delaney", "FreeSolv", "HIV", "Lipophilicity",
    "Malaria", "MOSES", "MUV", "OPV", "QM8", "QM9", "SIDER", "Tox21", "ToxCast",
    "USPTO50k", "ZINC250k",
    "ZINC2m", "PubChem110m",
    "FB15k", "FB15k237", "WN18", "WN18RR", "Hetionet",
    "Cora", "CiteSeer",
]
