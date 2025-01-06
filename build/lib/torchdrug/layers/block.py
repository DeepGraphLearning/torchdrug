from torch import nn
from torch.nn import functional as F

from torchdrug import layers


class ProteinResNetBlock(nn.Module):
    """
    Convolutional block with residual connection from `Deep Residual Learning for Image Recognition`_.

    .. _Deep Residual Learning for Image Recognition:
        https://arxiv.org/pdf/1512.03385.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        kernel_size (int, optional): size of convolutional kernel
        stride (int, optional): stride of convolution
        padding (int, optional): padding added to both sides of the input
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, activation="gelu"):
        super(ProteinResNetBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, stride, padding, bias=False)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, stride, padding, bias=False)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward(self, input, mask):
        """
        Perform 1D convolutions over the input.

        Parameters:
            input (Tensor): input representations of shape `(..., length, dim)`
            mask (Tensor): bool mask of shape `(..., length, dim)`
        """
        identity = input

        input = input * mask    # (B, L, d)
        out = self.conv1(input.transpose(1, 2)).transpose(1, 2)
        out = self.layer_norm1(out)
        out = self.activation(out)

        out = out * mask
        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
        out = self.layer_norm2(out)

        out += identity
        out = self.activation(out)

        return out


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention block from
    `Attention Is All You Need`_.

    .. _Attention Is All You Need:
        https://arxiv.org/pdf/1706.03762.pdf

    Parameters:
        hidden_dim (int): hidden dimension
        num_heads (int): number of attention heads
        dropout (float, optional): dropout ratio of attention maps
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(SelfAttentionBlock, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads))
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

    def forward(self, input, mask):
        """
        Perform self attention over the input.

        Parameters:
            input (Tensor): input representations of shape `(..., length, dim)`
            mask (Tensor): bool mask of shape `(..., length)`
        """
        query = self.query(input).transpose(0, 1)
        key = self.key(input).transpose(0, 1)
        value = self.value(input).transpose(0, 1)

        mask = (~mask.bool()).squeeze(-1)
        output = self.attn(query, key, value, key_padding_mask=mask)[0].transpose(0, 1)

        return output


class ProteinBERTBlock(nn.Module):
    """
    Transformer encoding block from
    `Attention Is All You Need`_.

    .. _Attention Is All You Need:
        https://arxiv.org/pdf/1706.03762.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        num_heads (int): number of attention heads
        attention_dropout (float, optional): dropout ratio of attention maps
        hidden_dropout (float, optional): dropout ratio of hidden features
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, hidden_dim, num_heads, attention_dropout=0,
                 hidden_dropout=0, activation="relu"):
        super(ProteinBERTBlock, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.hidden_dim = hidden_dim
        
        self.attention = SelfAttentionBlock(input_dim, num_heads, attention_dropout)
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.dropout1 = nn.Dropout(hidden_dropout)
        self.layer_norm1 = nn.LayerNorm(input_dim)

        self.intermediate = layers.MultiLayerPerceptron(input_dim, hidden_dim, activation=activation)

        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout2 = nn.Dropout(hidden_dropout)
        self.layer_norm2 = nn.LayerNorm(input_dim)
            
    def forward(self, input, mask):
        """
        Perform a BERT-block transformation over the input.

        Parameters:
            input (Tensor): input representations of shape `(..., length, dim)`
            mask (Tensor): bool mask of shape `(..., length)`
        """
        x = self.attention(input, mask)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.layer_norm1(x + input)

        hidden = self.intermediate(x)

        hidden = self.linear2(hidden)
        hidden = self.dropout2(hidden)
        output = self.layer_norm2(hidden + x)

        return output