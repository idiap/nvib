#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# For examples
import math

import torch
from einops import rearrange
from torch import Tensor
from torch._jit_internal import Optional, Tuple
from torch.nn.functional import linear, softmax
from torch.nn.init import xavier_uniform_

from nvib.denoising_attention import (
    denoising_attention_eval,
    denoising_attention_train,
    pytorch_scaled_dot_product_attention,
)

torch.manual_seed(42)

Ns, Nt, B, d, p = 5, 7, 2, 4, 4
nhead = 2

########## WE ASSUME LENGTH FIRST ##########

# key = value = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).float()
key = value = torch.rand(Ns, B, p)

# query = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).float()
query = torch.rand(Nt, B, p)

# Fixed from the pretrained model
in_proj_weight = torch.randn((3 * d, p))
in_proj_bias = torch.randn(3 * d)
xavier_uniform_(in_proj_weight)

w_q, w_k, w_v = in_proj_weight.split([d, d, d])
b_q, b_k, b_v = in_proj_bias.split([d, d, d])

q = linear(query, w_q, b_q)
k = linear(key, w_k, b_k)
v = linear(value, w_v, b_v)

dropout = 0

head_dim = int(d / nhead)

# Multihead and make batch first
mh_q = q.contiguous().view(Nt, B * nhead, head_dim).transpose(0, 1)
mh_k = k.contiguous().view(Ns, B * nhead, head_dim).transpose(0, 1)
mh_v = v.contiguous().view(Ns, B * nhead, head_dim).transpose(0, 1)

attn_mask = torch.zeros(B * nhead, 1, Ns)


# Old implementation that only used a single head (sh) and assumed p == d
def scaled_dot_product_attention_evaluation_sh(
    projected_u: Tensor,
    mu: Tensor,
    logvar: Tensor,
    pi: Tensor,
    w_v: Tensor,
    key_padding_mask: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """
    At evalutation time the denoising attention values are an interpolation between each mean and the query

    :param projected_u: [B,Nt,H]
    :param mu: [Ns,B,H]
    :param logvar: [Ns,B,H]
    :param pi: [Ns,B,1]
    :param w_v: [H,H] assuming square projection
    :param key_padding_mask:
    :param attn_mask: [B,1,Ns] broadcast over the Nt dimension
    :param dropout_p:
    :return:
    """

    # TODO: Multihead consideration here
    prior_var_u = math.sqrt(w_v.size(0))
    # Transform all in to [Bs,Ns,..]
    mu = mu.transpose(0, 1)
    pi = pi.transpose(0, 1)
    logvar = logvar.transpose(0, 1)

    var = torch.exp(logvar)
    biased_var = var + prior_var_u

    # where pi is zero include it to the attention mask
    pi_attn_mask = torch.zeros_like(pi.permute(0, 2, 1), dtype=torch.float)
    pi_attn_mask.masked_fill_(pi.permute(0, 2, 1).le(0), float("-inf"))
    attn_mask += pi_attn_mask

    # Attention term (masked by mu)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(projected_u, mu.permute(0, 2, 1) / biased_var.permute(0, 2, 1))

    #
    # (light breath) include bias terms
    #

    # Alpha term
    pi = torch.clamp(pi.clone(), min=1e-8)  # Make sure its above zero
    t1 = torch.log(pi).masked_fill_(key_padding_mask.permute(0, 2, 1), 0)  # [B,Ns,1]
    # L2 norm term
    t2 = 0.5 * ((torch.norm((mu / torch.sqrt(biased_var)), dim=-1)).unsqueeze(-1) ** 2)  # [B,Ns,1]
    # Variance penalty term
    t3 = torch.sum(
        torch.log(biased_var).masked_fill_(key_padding_mask.permute(0, 2, 1), 0), dim=-1
    ).unsqueeze(
        -1
    )  # [B,Ns,1]
    # # Bias terms into attention (broadcasted)
    attn = attn + (t1 - t2 - t3).permute(0, 2, 1)

    # Mask and softmax
    if attn_mask is not None:
        attn += attn_mask
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)

    # Interpolate attention between Z and projected query
    output = torch.bmm(attn, (var / biased_var)) * projected_u + torch.bmm(
        attn, ((prior_var_u / biased_var) * mu)
    )

    # Project into the correct space
    output = output @ w_v.T

    return output, attn


# A implementation using loops to make sure it make sense
def scaled_dot_product_attention_evaluation1attime(
    query: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    mu: Tensor,
    logvar: Tensor,
    pi: Tensor,
    w_v: Tensor,
    key_padding_mask: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    # TODO: Multihead consideration here
    prior_var_u = math.sqrt(w_k.size(0))  # root(d)
    # Transform all in to [Bs,Ns,..]
    mu = mu.transpose(0, 1)
    pi = pi.transpose(0, 1)
    logvar = logvar.transpose(0, 1)
    query = query.transpose(0, 1)

    var = torch.exp(logvar)
    biased_var = var + prior_var_u

    # Loop through the number of queries m
    for i in range(query.size(1)):
        query_i = query[:, i, :].unsqueeze(1)

        projected_u = linear(query_i, w_q) @ w_k

        # where pi is zero include it to the attention mask
        pi_attn_mask = torch.zeros_like(pi.permute(0, 2, 1), dtype=torch.float)
        pi_attn_mask.masked_fill_(pi.permute(0, 2, 1).le(0), float("-inf"))
        attn_mask += pi_attn_mask

        # Attention term (masked by mu)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(projected_u, mu.permute(0, 2, 1) / biased_var.permute(0, 2, 1))

        #
        # (light breath) include bias terms
        #

        # Alpha term
        pi = torch.clamp(pi.clone(), min=1e-8)  # Make sure its above zero
        t1 = torch.log(pi).masked_fill_(key_padding_mask.permute(0, 2, 1), 0)  # [B,Ns,1]
        # L2 norm term
        t2 = 0.5 * (
            (torch.norm((mu / torch.sqrt(biased_var)), dim=-1)).unsqueeze(-1) ** 2
        )  # [B,Ns,1]
        # Variance penalty term
        t3 = torch.sum(
            torch.log(biased_var).masked_fill_(key_padding_mask.permute(0, 2, 1), 0), dim=-1
        ).unsqueeze(
            -1
        )  # [B,Ns,1]
        # Bias terms into attention (broadcasted)
        attn = attn + (t1 - t2 - t3).permute(0, 2, 1)

        # Mask and softmax
        if attn_mask is not None:
            attn += attn_mask
        attn = softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = dropout(attn, p=dropout_p)

        # TODO: multihead this?
        # Interpolate attention between Z and projected query
        interpolation = (var / biased_var) * projected_u + (prior_var_u / biased_var) * mu
        current_output = torch.bmm(attn, interpolation)  # [B, i, p]
        try:
            output = torch.cat((output, current_output), dim=1)
            final_attn = torch.cat((final_attn, attn), dim=1)
        except:
            final_attn = attn
            output = current_output

    # Project into the correct space
    output = output @ w_v.T  # [B, m, d]

    return output, final_attn


###############################################################
# Tests
###############################################################


def test_denoising_mh_attention():
    """
    Is the true attention close to the training time denoising attention even for multihead attention
    """

    # Simple unrealistic initialisation such that they cancel out
    pi = torch.ones(B, Ns, nhead)
    Z = torch.zeros_like(key.transpose(0, 1))

    denoising_training_out, denoising_training_attn = denoising_attention_train(
        mh_q, mh_k, mh_v, pi, Z, attn_mask=attn_mask, dropout_p=dropout
    )
    attn_1, _ = pytorch_scaled_dot_product_attention(
        mh_q, mh_k, mh_v, attn_mask=attn_mask, dropout_p=dropout
    )

    # Is close is for the numerical precision of log pi
    assert torch.isclose(attn_1, denoising_training_out).all()


def test_denoising_attention_evaluation_1attime():
    """
    For a single head we test if the denoising evaluation looped version is equivalent to the non-looped version
    """
    nhead = 1
    head_dim = int(d / nhead)
    in_proj_bias = torch.zeros(3 * d)
    b_q, b_k, b_v = in_proj_bias.split([d, d, d])

    q = linear(query, w_q, b_q)

    dropout = 0

    head_dim = int(d / nhead)

    # Multihead and make batch first
    mh_q = q.contiguous().view(Nt, B * nhead, head_dim).transpose(0, 1)

    attn_mask = torch.zeros(B * nhead, 1, Ns)

    atol = 1e-4

    # Reshape the multihead query and weights
    mh_w_k = w_k.view(nhead, head_dim, -1)  # [heads, d/head, p]
    mh_w_v = w_v.view(nhead, head_dim, -1)  # [heads, d/head, p]
    mh_b_k = b_k.view(nhead, head_dim, -1)  # [heads, d/head, 1]
    mh_b_v = b_v.view(nhead, head_dim, -1)  # [heads, d/head, 1]
    q_reshape = mh_q.view(B, nhead, Nt, head_dim)  # [B, heads, Nt, d/head]

    # Project the multihead query into the p space from the e (d/head) space
    projected_u = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_w_k)
    projected_bias = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_b_k)

    mu = Z = key  # [Ns,B,P]

    # If the logvar is small then the attention is close to the training attention
    logvar = torch.ones_like(mu) * 0
    # Uniform probability over the source tokens
    pi = torch.ones(Ns, B, 1) / Ns

    # The pytorch implementation does this reshaping
    key_padding_mask = torch.zeros(B, 1, Ns, dtype=bool)

    denoising_test_out, denoising_test_attn = denoising_attention_eval(
        projected_u,
        projected_bias,
        mu,
        logvar,
        pi,
        mh_w_v,
        mh_b_v,
        key_padding_mask,
        attn_mask=attn_mask,
        dropout_p=dropout,
    )

    denoising_test_out1, denoising_test_attn1 = scaled_dot_product_attention_evaluation1attime(
        query,
        w_q,
        w_k,
        mu,
        logvar,
        pi,
        w_v,
        key_padding_mask,
        attn_mask=attn_mask,
        dropout_p=dropout,
    )

    # Assert that the training and test time functions are close
    assert (
        torch.isclose(denoising_test_out, denoising_test_out1, atol=atol).all()
        and torch.isclose(denoising_test_attn, denoising_test_attn1, atol=atol).all()
    )


def test_denoising_attention_evaluation_mh():
    """
    For a single head, and bias of zero, does the old version equal the new version which can do multihead
    """

    nhead = 1
    head_dim = int(d / nhead)
    in_proj_bias = torch.zeros(3 * d)
    b_q, b_k, b_v = in_proj_bias.split([d, d, d])

    q = linear(query, w_q, b_q)

    dropout = 0

    head_dim = int(d / nhead)

    # Multihead and make batch first
    mh_q = q.contiguous().view(Nt, B * nhead, head_dim).transpose(0, 1)

    attn_mask = torch.zeros(B * nhead, 1, Ns)

    # Reshape the multihead query and weights
    mh_w_k = w_k.view(nhead, head_dim, -1)  # [heads, d/head, p]
    mh_w_v = w_v.view(nhead, head_dim, -1)  # [heads, d/head, p]
    mh_b_k = b_k.view(nhead, head_dim, -1)  # [heads, d/head, 1]
    mh_b_v = b_v.view(nhead, head_dim, -1)  # [heads, d/head, 1]
    q_reshape = mh_q.view(B, nhead, Nt, head_dim)  # [B, heads, Nt, d/head]

    # Project the multihead query into the p space from the e (d/head) space
    projected_u = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_w_k)
    projected_bias = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_b_k)

    mu = Z = key  # [Ns,B,P]

    # If the logvar is small then the attention is close to the training attention
    logvar = torch.ones_like(mu) * 0

    # The pytorch implementation does this reshaping
    key_padding_mask = torch.zeros(B, 1, Ns, dtype=bool)
    # Uniform probability over the source tokens
    pi = torch.ones(Ns, B, 1) / Ns

    # The pytorch implementation does this reshaping
    key_padding_mask = torch.zeros(B, 1, Ns, dtype=bool)

    denoising_test_out, denoising_test_attn = scaled_dot_product_attention_evaluation_sh(
        projected_u.squeeze(1),
        mu,
        logvar,
        pi,
        w_v,
        key_padding_mask,
        attn_mask=attn_mask,
        dropout_p=dropout,
    )

    denoising_test_out2, denoising_test_attn2 = denoising_attention_eval(
        projected_u,
        projected_bias,
        mu,
        logvar,
        pi,
        mh_w_v,
        mh_b_v,
        key_padding_mask,
        attn_mask=attn_mask,
        dropout_p=dropout,
    )

    # Assert that the training and test time functions are close
    assert (
        torch.isclose(denoising_test_out, denoising_test_out2).all()
        and torch.isclose(denoising_test_attn, denoising_test_attn2).all()
    )


def test_denoising_attention_evaluation():
    """
    For multihead, when variance is 0, is the denosing attention training close to the test time
    """

    # Reshape the multihead query and weights
    mh_w_k = w_k.view(nhead, head_dim, -1)  # [heads, d/head, p]
    mh_w_v = w_v.view(nhead, head_dim, -1)  # [heads, d/head, p]
    mh_b_k = b_k.view(nhead, head_dim, -1)  # [heads, d/head, 1]
    mh_b_v = b_v.view(nhead, head_dim, -1)  # [heads, d/head, 1]

    q_reshape = mh_q.view(B, nhead, Nt, head_dim)  # [B, heads, Nt, d/head]

    # Project the multihead query into the p space from the e (d/head) space
    projected_u = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_w_k)
    projected_b = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_b_k)

    mu = Z = key.transpose(0, 1)

    # If the logvar is small then the attention is close to the training attention
    logvar = torch.ones_like(mu) * -200
    # Uniform probability over the source tokens
    pi = torch.ones(B, Ns, nhead) / Ns

    # The pytorch implementation does this reshaping
    key_padding_mask = torch.zeros(B * nhead, 1, Ns, dtype=bool)  # [B, Ns]

    breakpoint()
    denoising_training_out, denoising_training_attn = denoising_attention_train(
        mh_q, mh_k, mh_v, pi, Z, attn_mask=attn_mask, dropout_p=dropout
    )

    denoising_test_out, denoising_test_attn = denoising_attention_eval(
        projected_u,
        projected_b,
        mu,
        logvar,
        pi,
        mh_w_v,
        mh_b_v,
        key_padding_mask,
        attn_mask=attn_mask,
        dropout_p=dropout,
    )

    # Assert that the training and test time functions are close
    assert (
        torch.isclose(denoising_test_out, denoising_training_out).all()
        and torch.isclose(denoising_test_attn, denoising_training_attn).all()
    )


def test_multihead_and_reversing_multihead_query():
    # Go backward from multihead

    # Apply multihead
    reversed_q = mh_q.transpose(0, 1).contiguous().view(Nt, B, d)

    assert torch.isclose(q, reversed_q).all()


def test_multihead_weight_matrix_only():
    # apply mutihead to weight matrix
    w_q2 = w_q.view(nhead, head_dim, p)
    mh_b_q = b_q.view(nhead, head_dim).unsqueeze(0).unsqueeze(2)  # [1,heads,1, d/head]

    # torch.einsum("bnp, dp -> bnd",value, w_v) == v is equivalent
    mh_q_einsum = (
        (torch.einsum("nbp, hep -> bhne", query, w_q2) + mh_b_q)
        .contiguous()
        .view(B * nhead, Nt, head_dim)
    )

    assert torch.isclose(mh_q_einsum, mh_q).all()


def test_normal_multihead_with_einsum():
    # Q.K with einsum

    # Query x Key calculation in multihead
    mh_attn = torch.bmm(mh_q, mh_k.transpose(-2, -1))

    mh_attn_ein = torch.einsum("hme, hne -> hmn", mh_q, mh_k)

    assert torch.isclose(mh_attn, mh_attn_ein).all()


def test_einops_for_manipulations():
    # Apply multihead to query with einops
    # mh_q_ein = rearrange(q, "b n (e h) -> n (b h) e", h=nhead, e=head_dim).transpose(0, 1)
    mh_q_ein = rearrange(q, "n b (h e) -> (b h) n e", h=nhead, e=head_dim)

    assert torch.isclose(mh_q_ein, mh_q).all()


def test_multihead_attention_calculation():
    """
    Test multihead attention
    """

    # Query x Key calculation in multihead
    mh_attn = torch.bmm(mh_q, mh_k.transpose(-2, -1))

    # Multihead the kPy weight matrix
    mh_w_k = rearrange(w_k, "(h e) p -> h e p", h=nhead, e=head_dim)
    mh_b_k = rearrange(b_k, "(h e) -> h e 1", h=nhead, e=head_dim)

    # Reshape the multihead query
    mh_q_reshape = rearrange(mh_q, "(b h) m e -> b h m e", h=nhead, e=head_dim)

    # Project the multihead query into the p space from the e (d/head) space
    projected_u = torch.einsum("bhme, hep -> bhmp", mh_q_reshape, mh_w_k)
    projected_bias = torch.einsum("bhme, hep -> bhmp", mh_q_reshape, mh_b_k)

    # Project back from the p space to the Q.K attention matrix
    mh_attn_2 = (torch.einsum("bhmp, nbp -> bhmn", projected_u, key) + projected_bias).reshape(
        B * nhead, Nt, Ns
    )

    # print(torch.isclose(mh_attn, mh_attn_2))
    assert torch.isclose(mh_attn, mh_attn_2).all()


def test_multihead_v():
    # Query x Key calculation in multihead
    mh_attn = torch.bmm(mh_q, mh_k.transpose(-2, -1))
    mh_attn = softmax(mh_attn, dim=-1)
    out = torch.bmm(mh_attn, mh_v)

    # Reshape
    mh_w_v = w_v.view(nhead, head_dim, p)
    mh_b_v = b_v.view(nhead, head_dim, 1)
    mh_attn_reshape = mh_attn.view(B, nhead, Nt, Ns)

    # Project into p space by copying over heads
    out_p = torch.einsum("bhmn, nbp -> bhmp", mh_attn_reshape, value)
    # Project back into head space
    out_einsum = (
        torch.einsum("bhmp, hep -> bhme", out_p, mh_w_v).contiguous().view(B * nhead, Nt, head_dim)
    )
    out_bias = (
        torch.einsum("bhmp, hep -> bhme", mh_attn_reshape, mh_b_v)
        .contiguous()
        .view(B * nhead, Nt, head_dim)
    )

    out2 = out_einsum + out_bias

    assert torch.isclose(out, out2).all()


def main():
    pass


if __name__ == "__main__":
    main()
