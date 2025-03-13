#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Derive formulation for correct U space projection - assume B,Ns,H
#  torch.bmm(q, k.transpose(-2, -1)) == torch.bmm(((query@w_q.T).transpose(0,1)), ((key@w_k.T).!transpose(0,1)).transpose(-2, -1))
#                                    == torch.bmm(((query.transpose(0,1)@w_q.T)), ((key.transpose(0,1)@w_k.T)).transpose(-2, -1))
#                                    == (((query.transpose(0,1)@w_q.T)) @ ((key.transpose(0,1)@w_k.T)).transpose(-2, -1))
#                                    == (((query.transpose(0,1)@w_q.T)) @ w_k @ key.transpose(0,1).transpose(-2,-1))
#                                    == (((query.transpose(0,1)@w_q.T)) @ w_k @ key.permute(1,2,0))
#                                    == ((q @ w_k) @ key.permute(1,2,0))
#                                    == torch.bmm(q @ w_k, key.permute(1,2,0))

import math
import warnings
from typing import List

import torch
from torch import Tensor
from torch._jit_internal import Optional, Tuple
from torch.nn.functional import *
from torch.nn.functional import dropout, softmax
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules import Module
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter
from torch.overrides import handle_torch_function, has_torch_function


def make_causal_mask(B, Nt, Ns):
    # Makes a diagonal mask for the attention matrix
    causal_mask = torch.zeros(B, Nt, Ns)
    causal_mask_temp = ~torch.ones(B, Nt, Ns).tril(diagonal=0).bool()
    causal_mask = causal_mask.masked_fill_(causal_mask_temp, float("-inf"))

    return causal_mask


# This is a copy of the pytorch function in torch.nn.functional.scale_dot_product_attention
# Their implementation uses, C++, flash attention and memory efficient attention. which is
# better in almost all ways but this is comparable.
def pytorch_scaled_dot_product_attention(
    query, key, value, attn_mask, dropout_p, is_causal=False
):
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        query, key, value: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
        is_causal: if True, applies a causal mask to the attention weights.

    Shape:
        - query: :math:`(B, Nt, D)` where B is batch size, Nt is the target sequence length,
            and D is embedding dimension.
        - key: :math:`(B, Ns, D)` where B is batch size, Ns is the source sequence length,
            and D is embedding dimension.
        - value: :math:`(B, Ns, D)` where B is batch size, Ns is the source sequence length,
            and D is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
    """

    Nt = query.size(-2)
    Ns = key.size(-2)
    B = query.size(0)

    # Make causal mask if needed and Transform mask to 0s and -inf
    if is_causal:
        causal_mask = make_causal_mask(B, Nt, Ns)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.float().masked_fill_(attn_mask == True, float("-inf"))

    # Combine causal and attn mask
    attn_mask = attn_mask + causal_mask if is_causal else attn_mask

    # Calculate attention weights
    attn_weight = torch.softmax(
        (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1
    )
    # Apply dropout if specified
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight


def denoising_attention_train(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    pi: Tensor,
    Z: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal=False,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - pi: :math:`(B, Nl, 1)` where B is batch size, Nt is the target sequence length
        - Z: :math:`(B, Nl, p) where Ns is the source sequence length, B is batch size, and
            p is the embedding before projection and heads`

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """

    H, Nt, E = query.shape  # split over heads E is D / H
    B, Nl, P = Z.shape  # not split over heads
    nheads = int(P / E)

    # Make causal mask if needed and Transform mask to 0s and -inf
    if is_causal:
        causal_mask = make_causal_mask(B * nheads, Nt, Nl - 1)
        causal_mask = torch.cat(
            (torch.zeros((B * nheads, Nt, 1)).bool(), causal_mask), dim=-1
        )

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.float().masked_fill_(attn_mask == True, float("-inf"))

    # Combine causal and attn mask
    attn_mask = attn_mask + causal_mask if is_causal else attn_mask

    query = query / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(query, key.transpose(-2, -1))

    # Repeat pi over B dim for heads B1H1, ..., B1Hn, B2H1, ..., B2Hn, ...
    pi = torch.repeat_interleave(pi, nheads, dim=0)  # B, Nl, 1

    # where pi is zero include it to the attention mask
    pi_attn_mask = torch.zeros_like(pi.permute(0, 2, 1), dtype=torch.float)
    pi_attn_mask.masked_fill_(pi.permute(0, 2, 1).le(0), float("-inf"))
    attn_mask += pi_attn_mask

    # Include bias terms repeated over Nt dimension
    l2_norm = (1 / (2 * math.sqrt(E))) * ((torch.norm(Z, dim=-1)).unsqueeze(1) ** 2)
    l2_norm = torch.repeat_interleave(l2_norm, nheads, dim=0)

    pi = torch.clamp(pi.clone(), min=torch.finfo(pi.dtype).tiny).permute(
        0, 2, 1
    )  # Make sure its above zero

    attn += torch.log(pi) - l2_norm

    if attn_mask is not None:
        attn += attn_mask
    attn = softmax(attn, dim=-1)

    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, value)
    return output, attn


def denoising_attention_eval(
    projected_u: Tensor,
    projected_b: Tensor,
    mu: Tensor,
    logvar: Tensor,
    pi: Tensor,
    w_v: Tensor,
    b_v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal=False,
) -> Tuple[Tensor, Tensor]:
    """
    At evalutation time the denoising attention values are an interpolation between each mean and the query

    :param projected_u: [B,H,Nt,P] where H heads and P is the embedding dimension before QKV projection
    :param projected_b: [B,H,Nt,P] where H heads and P is the embedding dimension before QKV projection
    :param mu: [B,Nl,P]
    :param logvar: [B,Nl,P]
    :param pi: [B,Nl,1]
    :param w_v: [H,E, P]
    :param b_v: [B*H,1]
    :param attn_mask: [B,1,Nl] broadcast over the Nt dimension
    :param dropout_p:
    :return: attention values have shape :math:`(B, Nt, E)`;
             attention weights have shape :math:`(B, Nt, Ns)`
    """

    B, H, Nt, P = projected_u.shape
    B, Nl, P = mu.shape
    H, E, P = w_v.shape

    prior_var_u = math.sqrt(E)  # Keep all in d/h
    var = torch.exp(logvar)
    biased_var = var + prior_var_u

    # Make causal mask if needed and Transform mask to 0s and -inf
    if is_causal:
        causal_mask = make_causal_mask(B * H, Nt, Nl - 1)
        causal_mask = torch.cat(
            (torch.zeros((B * H, Nt, 1)).bool(), causal_mask), dim=-1
        )

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.float().masked_fill_(attn_mask == True, float("-inf"))

    # Repeat pi over B dim for heads B1H1, ..., B1Hn, B2H1, ..., B2Hn, ...
    pi = torch.repeat_interleave(pi, H, dim=0)  # B, Nl, 1

    # where pi is zero include it to the attention mask
    pi_attn_mask = torch.zeros_like(pi.permute(0, 2, 1), dtype=torch.float)
    pi_attn_mask.masked_fill_(pi.permute(0, 2, 1).le(0), float("-inf"))
    attn_mask += pi_attn_mask

    # Project back from the p space to the Q.K attention matrix
    attn = torch.einsum("bhmp, bnp -> bhmn", projected_u, mu / biased_var) + projected_b

    #
    # (light breath) include bias terms
    #
    # [B*H,1,Ns]
    # Alpha term repeated over each head
    pi = torch.clamp(
        pi.clone(), min=torch.finfo(pi.dtype).tiny
    )  # Make sure its above zero
    t1 = torch.log(pi).masked_fill_(
        attn_mask.permute(0, 2, 1) == float("-inf"), 0
    )  # [B*H,Ns,1]

    # L2 norm term repeated over each head
    t2 = 0.5 * ((torch.norm((mu / torch.sqrt(biased_var)), dim=-1)) ** 2).unsqueeze(-1)
    t2 = torch.repeat_interleave(t2, H, dim=0)  # [B*H,Ns,1]

    # Variance penalty term per head
    t3 = torch.log(torch.repeat_interleave(biased_var, H, dim=0))  # [B*H,Ns,P]
    t3 = t3.masked_fill_(attn_mask.permute(0, 2, 1) == float("-inf"), 0)
    t3 = torch.sum(0.5 * t3, dim=-1).unsqueeze(-1)  # [B*H,Ns,1]

    # Bias terms into attention copied over heads broadcasted over Nt
    attn = attn + (t1 - t2 - t3).view(B, H, Nl, 1).permute(0, 1, 3, 2)

    # Combine causal and attn mask
    attn_mask = (
        (attn_mask + causal_mask).view(B, H, Nt, Nl)
        if is_causal
        else attn_mask.view(B, H, 1, Nl)
    )

    # Mask and softmax
    if attn_mask is not None:
        attn += attn_mask
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)

    # Interpolate attention between Z and projected query
    output = torch.einsum(
        "bhmn, bnp -> bhmp", attn, (var / biased_var)
    ) * projected_u + torch.einsum(
        "bhmn, bnp -> bhmp", attn, ((prior_var_u / biased_var) * mu)
    )

    # Project into the correct space
    output = (
        (
            torch.einsum("bhmp, hep -> bhme", output, w_v)
            + torch.einsum("bhmp, hep -> bhme", attn, b_v)  # Add biases
        )
        .contiguous()
        .view(B * H, Nt, E)
    )

    return output, attn.contiguous().view(B * H, Nt, Nl)


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


# COPIED FROM torch.nn.functional
def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (
        Eq,
        Eq,
    ), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (
        Eq,
        Ek,
    ), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (
        Eq,
        Ev,
    ), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (
        Eq,
    ), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (
        Eq,
    ), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (
        Eq,
    ), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


# COPIED FROM torch.nn.module.activation
# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py
class DenoisingMultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DenoisingMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            self.k_proj_weight = Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(DenoisingMultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
                when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
                and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
                key-value pairs to produce the output. See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
                ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
                :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
                ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
                :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
                value will be ignored.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
              :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
              the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
            - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
              size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
              when ``need_weights=True``.
        """
        if self.batch_first:

            (key, pi, mu, logvar) = key
            (value, pi, mu, logvar) = value
            # Transpose each tensor and put back in tuple
            key = (
                key.transpose(1, 0),
                pi.transpose(1, 0),
                mu.transpose(1, 0),
                logvar.transpose(1, 0),
            )
            value = (
                value.transpose(1, 0),
                pi.transpose(1, 0),
                mu.transpose(1, 0),
                logvar.transpose(1, 0),
            )
            query = query.transpose(1, 0)

            # query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = denoising_multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
            )
        else:
            attn_output, attn_output_weights = denoising_multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
            )
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def denoising_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    # Fetch the variables needed
    (key, pi, mu, logvar) = key
    (value, _, _, _) = value

    tens_ops = (
        query,
        key,
        value,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        out_proj_weight,
        out_proj_bias,
    )
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert (
            q_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            k_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            v_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert (
                attn_mask.is_floating_point() or attn_mask.dtype == torch.bool
            ), f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_k.size(0) == bsz * num_heads
        ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert (
            static_k.size(2) == head_dim
        ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_v.size(0) == bsz * num_heads
        ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert (
            static_v.size(2) == head_dim
        ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # Reminder:
    # k == v [B*H,Ns,D/H]
    # q [B*H,Nt,D/H]
    # key == value [Ns,B,P]
    # query [Nt,B,P]

    # pi [Ns,B,1]
    # mu [Ns,B,P]
    # logvar [Ns,B,P]

    if training:
        #
        # (deep breath) calculate attention and out projection
        #

        # Denoising attention needs a attn_mask of shape [B,Nt,Ns]

        # [B*H,Nt, D/H]
        attn_output, attn_output_weights = denoising_attention_train(
            q, k, v, pi.transpose(0, 1), key.transpose(0, 1), attn_mask, dropout_p
        )

        # Reverse multiheads
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )  # [Nt,B,D]

    else:
        # TRAINING TIME FUNCTION
        # attn_output, attn_output_weights = scaled_dot_product_attention_training(q, k, v,
        #                                                                          pi, key,
        #                                                                          attn_mask,
        #                                                                          dropout_p)

        # Get weights
        _, w_k, w_v = in_proj_weight.split([embed_dim, embed_dim, embed_dim])
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)

        # Reshape the multihead query and weights
        mh_w_k = w_k.view(num_heads, head_dim, -1)  # [heads, d/head, p]
        mh_w_v = w_v.view(num_heads, head_dim, -1)  # [heads, d/head, p]
        mh_b_v = b_v.view(num_heads, head_dim).unsqueeze(-1)  # [heads, d/head, 1]
        mh_b_k = b_k.view(num_heads, head_dim).unsqueeze(-1)  # [heads, d/head, 1]
        q_reshape = q.view(bsz, num_heads, tgt_len, head_dim)  # [B, heads, Nt, d/head]

        # Project the multihead query and bias into the p space from the e (d/head) space
        projected_u = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_w_k)
        projected_bias = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_b_k)

        breakpoint()
        attn_output, attn_output_weights = denoising_attention_eval(
            projected_u,
            projected_bias,
            mu,
            logvar,
            pi,
            mh_w_v,
            mh_b_v,
            key_padding_mask,
            attn_mask,
            dropout_p,
        )

        # Reverse multiheads
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )  # [Nt,B,D]

    # output projection
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        # Average attention weights over heads
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
        # Return all attention weights
        else:
            return attn_output, attn_output_weights
    else:
        return attn_output, None
