import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.attention as attn
from torch.nn import init
import math
# from lightning.pytorch.cli import instantiate_class
from typing import Optional, Mapping, Sequence, Union
# from .util import get_rank, safe_gather

TNonLin = Optional[Union[str,Mapping]]

def apply_p_drop(m:nn.Module, p:float):
    def set_p_drop(m:nn.Module):
        if isinstance(m, nn.Dropout):
            m.p = p
    m.apply(set_p_drop)

class StackedLinear(torch.nn.Module):
    def __init__(self, d_in:int, d_out:int, n_stack:int, bias:bool=True,
                 device=None, dtype=None, transpose=False) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_stack = n_stack
        self.transpose = transpose
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((n_stack, d_in, d_out), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(n_stack, 1, d_out, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Lifted from torch.nn.Linear
        # NB: this is important; you could get really weird results (loss off the scale) if you get this wrong
        for i in range(self.n_stack):
            init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            for i in range(self.n_stack):
                init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight        
        if self.bias is not None:
            out += self.bias
        if self.transpose:
            out = out.transpose(0,1)
        return out
        
    def extra_repr(self) -> str:
        return 'd_in={}, d_out={}, n_stack={}, bias={}'.format(
            self.d_in, self.d_out, self.n_stack, self.bias is not None
        )    


class ClsHeadBlock(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, n_stack:Optional[int], layer_norm: bool, p_drop:Optional[float],
                 nonlinearity:TNonLin, bias:bool) -> None:
        super().__init__()
        layer = StackedLinear(in_dim, out_dim, n_stack, bias=bias) if n_stack is not None else nn.Linear(in_dim, out_dim, bias=bias)
        self.layers = nn.Sequential(layer)
        if layer_norm:
            self.layers.append(nn.LayerNorm(out_dim))
        if p_drop is not None and p_drop != 0:
            self.layers.append(nn.Dropout(p_drop))
        if nonlinearity is not None:
            nonlin = getattr(nn, nonlinearity)()
            self.layers.append(nonlin)
    
    def forward(self, x):
        return self.layers(x)


class StackedClsHead(nn.Module):

    def __init__(self, embedding_dim:int, n_stack:int, layers:Optional[Mapping[str, int]]=None,
                 layer_norm:bool=False, p_drop:Optional[float]=None, 
                 nonlinearity:TNonLin=None, bias:bool=False, end_bias:bool=False) -> None:
        super().__init__()
        layer_dims = []
        if layers:
            layer_dims = [round(layers['d'])] * round(layers['n'])
        layer_dims = [embedding_dim] + layer_dims
        self.layers = nn.Sequential(*[ClsHeadBlock(layer_dims[i], layer_dims[i+1], n_stack,
                                                   layer_norm, p_drop, nonlinearity, bias) 
                                    for i in range(len(layer_dims) - 1)])
        self.layers.append(StackedLinear(layer_dims[-1], 2, n_stack, bias=end_bias, transpose=True)) # todo p_drop

    def forward(self, x):
        return self.layers(x)
    

class ClsHead(nn.Module):

    def __init__(self, embedding_dim:int, layers:Optional[Mapping[str, int]]=None,
                 layer_norm:bool=False, p_drop:Optional[float]=None, 
                 nonlinearity:TNonLin=None, bias:bool=False, end_bias:bool=False,
                 n_stack:Optional[int]=None, out_dim:int=2) -> None:
        super().__init__()
        layer_dims = []
        if layers:
            layer_dims = [round(layers['d'])] * round(layers['n'])
        layer_dims = [embedding_dim] + layer_dims
        self.layers = nn.Sequential(*[ClsHeadBlock(layer_dims[i], layer_dims[i+1], n_stack,
                                                   layer_norm, p_drop, nonlinearity, bias) 
                                    for i in range(len(layer_dims) - 1)])
        self.layers.append(ClsHeadBlock(layer_dims[-1], out_dim, n_stack=None, 
                                        layer_norm=layer_norm, p_drop=p_drop, nonlinearity=None, bias=end_bias))
        
    def forward(self, x):
        return self.layers(x)
    
class SetTransformerPooling(nn.Module):
    def __init__(self, d:int, num_heads:int, seq_dim:int|None=None, num_seeds:int=1, layer_norm:bool=False, p_drop:float=.0, keep_dim:bool=False,
                 can_use_efficient:bool=True) -> None:
        """
        [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. Lee et al, ICLM 2019](https://arxiv.org/pdf/1810.00825v3.pdf)
        Section 3.2
        """
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, num_seeds, d))
        nn.init.xavier_uniform_(self.seed)
        if seq_dim is None:
            seq_dim = d
        self.mha = nn.MultiheadAttention(d, num_heads, dropout=p_drop, batch_first=True, kdim=seq_dim, vdim=seq_dim)
        self.layer_norm = nn.LayerNorm(d) if layer_norm else None
        self.keep_dim = keep_dim
        self.attn_flags = [attn.SDPBackend.FLASH_ATTENTION, attn.SDPBackend.MATH, attn.SDPBackend.CUDNN_ATTENTION]
        if can_use_efficient:
            self.attn_flags.append(attn.SDPBackend.EFFICIENT_ATTENTION)

    def forward(self, x:torch.Tensor, mask:Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        if mask is not None:
            mask = ~(mask.to(dtype=bool))
        seed = self.seed.expand(x.shape[0], -1, -1)
        # Required for EMLC to work.
        # Forward mode differentiation is not implemented for memory efficient attention, 
        # so need to disable optimization here. NB: this might change in future version of PyTorch
        with attn.sdpa_kernel(self.attn_flags):
            out, _ = self.mha(seed, x, x, key_padding_mask=mask, need_weights=False)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        if not self.keep_dim:
            out = out.flatten(start_dim=1)
        return out
    
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor, *, mask:torch.Tensor, **kwargs) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        mask = mask.unsqueeze(-1)
        batch_seq_len = mask.sum(dim=1)
        x_sum = x.masked_fill(~mask, 0).sum(dim=1)
        return x_sum / batch_seq_len
    
class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor, *, mask:torch.Tensor, **kwargs) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        mask = mask.unsqueeze(-1)
        x = x.masked_fill(~mask, float('-inf'))
        ret, _ = x.max(dim=1)
        return ret
