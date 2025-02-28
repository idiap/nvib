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

import torch
from torch import Tensor
from torch._jit_internal import Optional, Tuple
from torch.nn.functional import _canonical_mask, dropout, softmax


def make_causal_mask(B, Nt, Ns):
    # Makes a diagonal mask for the attention matrix
    causal_mask = torch.zeros(B, Nt, Ns)
    causal_mask_temp = ~torch.ones(B, Nt, Ns).tril(diagonal=0).bool()
    causal_mask = causal_mask.masked_fill_(causal_mask_temp, float("-inf"))

    return causal_mask


# This is a copy of the pytorch function in torch.nn.functional.scale_dot_product_attention
# Their implementation uses, C++, flash attention and memory efficient attention. which is
# better in almost all ways but this is comparable.
def pytorch_scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal=False):
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
        causal_mask = torch.cat((torch.zeros((B * nheads, Nt, 1)).bool(), causal_mask), dim=-1)

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
        causal_mask = torch.cat((torch.zeros((B * H, Nt, 1)).bool(), causal_mask), dim=-1)

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
    pi = torch.clamp(pi.clone(), min=torch.finfo(pi.dtype).tiny)  # Make sure its above zero
    t1 = torch.log(pi).masked_fill_(attn_mask.permute(0, 2, 1) == float("-inf"), 0)  # [B*H,Ns,1]

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
        (attn_mask + causal_mask).view(B, H, Nt, Nl) if is_causal else attn_mask.view(B, H, 1, Nl)
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
    ) * projected_u + torch.einsum("bhmn, bnp -> bhmp", attn, ((prior_var_u / biased_var) * mu))

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
