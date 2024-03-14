from typing import Optional, Tuple, List, Callable

import torch
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import relu
from torch.nn.modules.activation import MultiheadAttention

def assert_padding_type(src_key_padding_mask: Optional[Tensor]):
    if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
            
            
class AttentionStoreTransformerEncoderLayer(TransformerEncoderLayer):
    """Should be identical to TransformerEncoderLayer, except that it stores the attention weights."""
    __constants__ = ['batch_first', 'norm_first']

    # def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
    #              activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
    #              layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
    #              device=None, dtype=None) -> None:
    #     factory_kwargs = {'device': device, 'dtype': dtype}
    #     super(TransformerEncoderLayer, self).__init__()
    #     self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
    #                                         **factory_kwargs)
    #     # Implementation of Feedforward model
    #     self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
    #     self.dropout = Dropout(dropout)
    #     self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

    #     self.norm_first = norm_first
    #     self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    #     self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    #     self.dropout1 = Dropout(dropout)
    #     self.dropout2 = Dropout(dropout)

    #     # Legacy string support for activation function.
    #     if isinstance(activation, str):
    #         activation = _get_activation_fn(activation)

    #     # We can't test self.activation in forward() in TorchScript,
    #     # so stash some information about it instead.
    #     if activation is F.relu or isinstance(activation, torch.nn.ReLU):
    #         self.activation_relu_or_gelu = 1
    #     elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
    #         self.activation_relu_or_gelu = 2
    #     else:
    #         self.activation_relu_or_gelu = 0
    #     self.activation = activation

    # def __setstate__(self, state):
    #     super(TransformerEncoderLayer, self).__setstate__(state)
    #     if not hasattr(self, 'activation'):
    #         self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        assert_padding_type(src_key_padding_mask)

        x = src
        if self.norm_first:
            sa, attention_weight = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + sa
            x = x + self._ff_block(self.norm2(x))
        else:
            sa, attention_weight = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + sa)
            x = self.norm2(x + self._ff_block(x))

        return x, attention_weight

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        x, attention = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        return self.dropout1(x), attention

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class AttentionStoreTransformerEncoder(TransformerEncoder):
    r"""Should be identical to TransformerEncoder, except that it stores the attention weights."""

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, List[Tensor]]:
        assert_padding_type(src_key_padding_mask)


        output = src
        src_key_padding_mask_for_layers = src_key_padding_mask

        attention_layers = []
        for mod in self.layers:
            output, attention = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)
            attention_layers.append(attention)            

        if self.norm is not None:
            output = self.norm(output)

        return output, attention_layers


            
class AttentionInjectLayer(TransformerEncoderLayer):
    """Attention layer which let you inject a second matrix along the K seq dimension."""
    def forward(self, src: Tensor, inject: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        assert_padding_type(src_key_padding_mask)

        x = src
        x_inject = torch.cat((src, inject))
        if self.norm_first:
            sa = self._sa_block(self.norm1(x), self.norm1(x_inject), src_mask, src_key_padding_mask)
            x = x + sa
            x = x + self._ff_block(self.norm2(x))
        else:
            sa = self._sa_block(x, x_inject, src_mask, src_key_padding_mask)
            x = self.norm1(x + sa)
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x: Tensor, x_inject: Tensor, attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        x, _ = self.self_attn(x, x_inject, x_inject,  # Q, K, V
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class AttentionStoreEmbeddingsTransformerEncoder(TransformerEncoder):
    """Transformer which store and return all embeddings - No need of special attention layer."""

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, List[Tensor]]:
        assert_padding_type(src_key_padding_mask)
        assert not isinstance(self.layers[0], AttentionInjectLayer)

        output = src
        src_key_padding_mask_for_layers = src_key_padding_mask

        embedding_layers = [output.detach().clone()]
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)
            embedding_layers.append(output.detach().clone())

        if self.norm is not None:
            output = self.norm(output)

        return output, embedding_layers


class AttentionInjectEmbeddingsTransformerEncoder(TransformerEncoder):
    """Transformer which let you inject input from one self encoder to a second one along K, V dimension.
    Need of an Attention Inject layers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.layers[0], AttentionInjectLayer)
    
    def forward(self, src: Tensor, embeddings: List[Tensor],
                mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        assert_padding_type(src_key_padding_mask)

        output = src
        src_key_padding_mask_for_layers = src_key_padding_mask

        for mod, embedding_inject in zip(self.layers, embeddings):
            output = mod(output, inject=embedding_inject, src_mask=mask,
                          src_key_padding_mask=src_key_padding_mask_for_layers)

        if self.norm is not None:
            output = self.norm(output)

        return output


class SymetricInjectLayer(TransformerEncoderLayer):
    """Attention layer, containing self attention (one or two layers)
    And can concatenate output of another layer into either any or all K, Q, V matrices.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float=0.1,
                second_attention: bool=False, batch_first: bool=False,
                device=None, dtype=None, **kwargs) -> None:
        super().__init__(d_model, nhead, dropout=dropout, batch_first=batch_first,
                          device=device, dtype=dtype, **kwargs)

        self.second_attention = second_attention
        if second_attention:
            self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout,
                                                batch_first=batch_first, device=device, dtype=dtype)
            torch.nn.init.zeros_(self.self_attn2.in_proj_weight)
            torch.nn.init.zeros_(self.self_attn2.out_proj.weight)

    def forward(self, src: Tensor, inject: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        assert_padding_type(src_key_padding_mask)

        x = src
        
        if self.second_attention:
            num_tokens = len(x)
            first_padding_mask = src_key_padding_mask[:, :num_tokens] if src_key_padding_mask is not None else None
            sa = self._sa_block(x, x, src_mask, first_padding_mask, self.self_attn2)
            x = self.norm1(x + sa)

        x_inject = torch.cat((x, inject))

        sa = self._sa_block(x, x_inject, src_mask, src_key_padding_mask, self.self_attn)
        x = self.norm1(x + sa)

        x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x: Tensor, x_inject: Tensor, attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor], atention_func: Callable) -> Tensor:
        x, _ = atention_func(x, x_inject, x_inject,  # Q, K, V
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class DoubleInjectTransformerEncoder(TransformerEncoder):
    """Running the same transformer twice, while injecting K, Q, V matrices from one inference to anoter."""

    def forward(self, actor1: Tensor, actor2: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor]:
        assert_padding_type(src_key_padding_mask)
        assert isinstance(self.layers[0], SymetricInjectLayer)

        embeding_a, embeding_b = actor1, actor2
        for layer in self.layers:
            embeding = layer(embeding_a, embeding_b, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            embeding_b = layer(embeding_b, embeding_a, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

            embeding_a = embeding

        # if self.norm is not None:
        #     output = self.norm(output)

        return embeding_a, embeding_b
