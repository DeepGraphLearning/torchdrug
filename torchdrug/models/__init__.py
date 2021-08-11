from .chebnet import ChebyshevConvolutionalNetwork
from .gcn import GraphConvolutionalNetwork, RelationalGraphConvolutionalNetwork
from .gat import GraphAttentionNetwork
from .gin import GraphIsomorphismNetwork
from .schnet import SchNet
from .mpnn import MessagePassingNeuralNetwork
from .neuralfp import NeuralFingerprint
from .infograph import InfoGraph
from .flow import GraphAutoregressiveFlow
from .embedding import TransE, DistMult, ComplEx, RotatE, SimplE
from .neurallp import NeuralLogicProgramming
from .kbgat import KnowledgeBaseGraphAttentionNetwork


# alias
ChebNet = ChebyshevConvolutionalNetwork
GCN = GraphConvolutionalNetwork
GAT = GraphAttentionNetwork
RGCN = RelationalGraphConvolutionalNetwork
GIN = GraphIsomorphismNetwork
MPNN = MessagePassingNeuralNetwork
NFP = NeuralFingerprint
GraphAF = GraphAutoregressiveFlow
NeuralLP = NeuralLogicProgramming
KBGAT = KnowledgeBaseGraphAttentionNetwork

__all__ = [
    "ChebyshevConvolutionalNetwork", "GraphConvolutionalNetwork", "RelationalGraphConvolutionalNetwork",
    "GraphAttentionNetwork", "GraphIsomorphismNetwork", "SchNet", "MessagePassingNeuralNetwork",
    "NeuralFingerprint",
    "InfoGraph",
    "GraphAutoregressiveFlow",
    "TransE", "DistMult", "ComplEx", "RotatE", "SimplE",
    "NeuralLogicProgramming", "KnowledgeBaseGraphAttentionNetwork",
    "ChebNet", "GCN", "GAT", "RGCN", "GIN", "MPNN", "NFP",
    "GraphAF", "NeuralLP", "KBGAT",
]