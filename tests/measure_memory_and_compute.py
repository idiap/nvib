# Set directory one higher
import os.path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import math

import torch
import torch.nn as nn
from torch.nn.functional import linear, scaled_dot_product_attention, softmax
from torch.nn.init import xavier_uniform_

from nvib.denoising_attention import (  # DenoisingMultiheadAttention,
    MultiheadAttention2,
    MultiheadAttention3,
    denoising_attention_eval,
    denoising_attention_train,
)
from nvib.nvib_layer import Nvib


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def my_scaled_dot_product_attention(Q, K, V, attn_mask, dropout_p, is_causal=False):
    """
    Exactly the same as scaled_dot_product_attention but returns the attention weights

    Args:
        Q (tensor): query tensor of shape (L, B, P)
        K (tensor): key tensor of shape (S, B, P)
        V (tensor): value tensor of shape (S, B, P)
        attn_mask (tensor): Attention mask boolean of shape (L, S)
        dropout_p (float?): Dropout probability
        is_causal (bool, optional): Causal mask flag for attention. Defaults to False.

    Returns:
        tensor: Attention output of shape (L, B, P)
        tensor: Attention weights of shape (L, S)
    """
    L = Q.size(-2)
    S = K.size(-2)
    attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
    attn_mask = (
        attn_mask.masked_fill(not attn_mask, -float("inf"))
        if attn_mask.dtype == torch.bool
        else attn_mask
    )
    attn_weight = torch.softmax(
        (Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1
    )
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ V, attn_weight


########################################################################################
# Test Nvib layer
########################################################################################

torch.manual_seed(42)

# Set parameter sizes
Ns, Nt, B, P, d = 10, 6, 2, 32, 32
nhead = 1
kappa = 1

# Define inputs and masks
number_samples = 3
encoder_output = torch.rand(Ns, B, P)  # Key and value
src_key_padding_mask = torch.zeros((B, Ns), dtype=bool)
tgt = torch.rand(Nt, B, P)  # query
tgt_key_padding_mask = torch.zeros((B, Nt), dtype=bool)
memory_key_padding_mask = torch.zeros((number_samples, Ns), dtype=bool)
device = "cpu"
dropout = 0
head_dim = int(d / nhead)

# Define Nvib layer
nvib_layer = Nvib(
    size_in=P,
    size_out=P,
    prior_mu=None,
    prior_var=None,
    prior_alpha=None,
    delta=1,
    kappa=kappa,
    nheads=nhead,
    mu_tau=None,
    alpha_tau=None,  # Initalised to identical
    stdev_tau=None,  # Initalised to identical
)
nvib_layer.train()
latent_dict = nvib_layer(encoder_output, src_key_padding_mask)


# Fixed from the pretrained model
in_proj_weight = torch.randn((3 * d, P))
in_proj_bias = torch.randn(3 * d)
xavier_uniform_(in_proj_weight)

w_q, w_k, w_v = in_proj_weight.split([d, d, d])
b_q, b_k, b_v = in_proj_bias.split([d, d, d])

q = linear(tgt, w_q, b_q)
k = linear(encoder_output, w_k, b_k)
v = linear(encoder_output, w_v, b_v)

# Denoising has an extra dimension
d_k = linear(latent_dict["z"][0], w_k, b_k)
d_v = linear(latent_dict["z"][0], w_v, b_v)

# Multihead and make batch first
mh_q = q.contiguous().view(Nt, B * nhead, head_dim).transpose(0, 1)
mh_k = k.contiguous().view(Ns, B * nhead, head_dim).transpose(0, 1)
mh_v = v.contiguous().view(Ns, B * nhead, head_dim).transpose(0, 1)

# Multihead and make batch first (denoising)
d_mh_v = d_v.contiguous().view((Ns + 1) * kappa, B * nhead, head_dim).transpose(0, 1)
d_mh_k = d_k.contiguous().view((Ns + 1) * kappa, B * nhead, head_dim).transpose(0, 1)


attn_mask = torch.zeros(B * nhead, 1, Ns)
d_attn_mask = torch.zeros(B * nhead, 1, (Ns + 1) * kappa)

# Define Transformer decoder
# decoder_layer = nn.TransformerDecoderLayer(
#     d_model=d, dim_feedforward=4 * d, nhead=nhead, dropout=dropout
# )
# transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

# Replace MultiheadAttention with DenoisingMultiheadAttention
# for layer_num, layer in enumerate(transformer_decoder.layers):
#     layer.multihead_attn = DenoisingMultiheadAttention(
#         embed_dim=d, num_heads=nhead, dropout=dropout, bias=True
#     )

# Run Transformer decoder
# output = transformer_decoder(
#     tgt=tgt,
#     memory=latent_dict["z"],
#     tgt_key_padding_mask=tgt_key_padding_mask,
#     memory_key_padding_mask=latent_dict["memory_key_padding_mask"],
# )


def test_nvib_layer_initialisations():
    """
    Given the initialisation of the NVIB layer, the encoder output should be equal to the mu
    and the logvar should be zero. Only when the bias is initialised to zero,  we get commented
    out alpha == exp_l2_norm
    """

    d = encoder_output.shape[-1]
    mu = latent_dict["mu"]
    logvar = latent_dict["logvar"]
    alpha = latent_dict["alpha"]

    l2_norm = ((torch.norm(mu, dim=-1) ** 2) / (2 * math.sqrt(d))).unsqueeze(-1)
    exp_l2_norm = torch.exp(l2_norm)

    # The encoder output == mu
    # The encoder variance is zero
    # The alpha is the exponential of the l2 norm (only when bias is initialised to zero)
    assert (
        torch.allclose(mu[1:, :, :], encoder_output)
        and torch.allclose(torch.exp(logvar[1:, :, :]), torch.zeros_like(encoder_output))
        # and torch.allclose(alpha, exp_l2_norm)
    )


def test_softmax_nvib_initialisation():
    """
    When normalised by softmax, the transformer output, and nvib at training function should be the same
    We need to put the nvib in eval mode as the sample through dirichlet still changes even with large alphas
    """

    d_out, d_attn = denoising_attention_train(
        mh_q,
        d_mh_k,
        d_mh_v,
        latent_dict["pi"],
        latent_dict["z"][0],
        attn_mask=d_attn_mask,
        dropout_p=dropout,
    )

    out, attn = my_scaled_dot_product_attention(
        mh_q, mh_k, mh_v, attn_mask=attn_mask, dropout_p=dropout
    )

    # Is close is for the numerical precision of log pi
    assert torch.allclose(attn, d_attn[:, :, 1:]) and torch.allclose(out, d_out)


def test_softmax_nvib_initialisation_eval():
    """
    When nvib layer is in eval model given the initialisation the denoising attentions are equal
    """

    nvib_layer = Nvib(
        size_in=P,
        size_out=P,
        prior_mu=None,
        prior_var=None,
        prior_alpha=None,
        delta=1,
        kappa=kappa,
        nheads=nhead,
        alpha_tau=None,
        mu_tau=None,
        stdev_tau=None,
    )
    nvib_layer.eval()

    latent_dict = nvib_layer(encoder_output, src_key_padding_mask)
    # Reshape the multihead query and weights
    mh_w_k = w_k.view(nhead, head_dim, -1)  # [heads, d/head, p]
    mh_w_v = w_v.view(nhead, head_dim, -1)  # [heads, d/head, p]
    mh_b_k = b_k.view(nhead, head_dim, -1)  # [heads, d/head, 1]
    mh_b_v = b_v.view(nhead, head_dim, -1)  # [heads, d/head, 1]

    q_reshape = mh_q.view(B, nhead, Nt, head_dim)  # [B, heads, Nt, d/head]

    # Project the multihead query into the p space from the e (d/head) space
    projected_u = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_w_k)
    projected_b = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_b_k)

    d_out, d_attn = denoising_attention_train(
        mh_q,
        d_mh_k,
        d_mh_v,
        latent_dict["pi"],
        latent_dict["z"][0],
        attn_mask=d_attn_mask,
        dropout_p=dropout,
    )

    dt_out, dt_attn = denoising_attention_eval(
        projected_u,
        projected_b,
        latent_dict["mu"],
        latent_dict["logvar"],
        latent_dict["pi"],
        mh_w_v,
        mh_b_v,
        latent_dict["memory_key_padding_mask"].repeat(nhead, 1).unsqueeze(1),  # [B*H, 1, Nl]
        attn_mask=d_attn_mask,
        dropout_p=dropout,
    )

    # During eval the priors are marginally different when softmaxed - due to the variance penalty term

    assert torch.allclose(d_out, dt_out) and torch.allclose(d_attn[:, :, 1:], dt_attn[:, :, 1:])


def test_softmax_nvib_initialisation_train():
    """
    When the nvib layer is in train mode, the denoising attentions are equal
    """

    nvib_layer = Nvib(
        size_in=P,
        size_out=P,
        prior_mu=None,
        prior_var=None,
        prior_alpha=None,
        delta=1,
        kappa=kappa,
        nheads=nhead,
        alpha_tau=None,
        mu_tau=None,
        stdev_tau=None,
    )
    nvib_layer.train()

    latent_dict = nvib_layer(encoder_output, src_key_padding_mask)
    # Reshape the multihead query and weights
    mh_w_k = w_k.view(nhead, head_dim, -1)  # [heads, d/head, p]
    mh_w_v = w_v.view(nhead, head_dim, -1)  # [heads, d/head, p]
    mh_b_k = b_k.view(nhead, head_dim, -1)  # [heads, d/head, 1]
    mh_b_v = b_v.view(nhead, head_dim, -1)  # [heads, d/head, 1]

    q_reshape = mh_q.view(B, nhead, Nt, head_dim)  # [B, heads, Nt, d/head]

    # Project the multihead query into the p space from the e (d/head) space
    projected_u = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_w_k)
    projected_b = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_b_k)

    d_out, d_attn = denoising_attention_train(
        mh_q,
        d_mh_k,
        d_mh_v,
        latent_dict["pi"],
        latent_dict["z"][0],
        attn_mask=d_attn_mask,
        dropout_p=dropout,
    )

    dt_out, dt_attn = denoising_attention_eval(
        projected_u,
        projected_b,
        latent_dict["mu"],
        latent_dict["logvar"],
        latent_dict["pi"],
        mh_w_v,
        mh_b_v,
        latent_dict["memory_key_padding_mask"].repeat(nhead, 1).unsqueeze(1),  # [B*H, 1, Nl]
        attn_mask=d_attn_mask,
        dropout_p=dropout,
    )

    assert torch.allclose(d_out, dt_out) and torch.allclose(d_attn, dt_attn)


def test_multisample():
    """
    When we take kappa samples the sum pi sum should be 1.
    """

    kappa = 10

    nvib_layer = Nvib(
        size_in=P,
        size_out=P,
        prior_mu=None,
        prior_var=None,
        prior_alpha=None,
        delta=1,
        kappa=kappa,
        nheads=nhead,
        alpha_tau=None,
        mu_tau=None,
        stdev_tau=None,
    )
    nvib_layer.train()

    latent_dict = nvib_layer(encoder_output, src_key_padding_mask)

    sum_pi = torch.sum(latent_dict["pi"], dim=0)
    assert torch.allclose(sum_pi, torch.ones_like(sum_pi))


def main():
    # Import all necessary libraries
    from torch.profiler import ProfilerActivity, profile, record_function

    nvib_layer = Nvib(
        size_in=P,
        size_out=P,
        prior_mu=None,
        prior_var=None,
        prior_alpha=None,
        delta=1,
        kappa=kappa,
        nheads=nhead,
        alpha_tau=None,
        mu_tau=None,
        stdev_tau=None,
    )
    nvib_layer.train()

    latent_dict = nvib_layer(encoder_output, src_key_padding_mask)
    # Reshape the multihead query and weights
    mh_w_k = w_k.view(nhead, head_dim, -1)  # [heads, d/head, p]
    mh_w_v = w_v.view(nhead, head_dim, -1)  # [heads, d/head, p]
    mh_b_k = b_k.view(nhead, head_dim, -1)  # [heads, d/head, 1]
    mh_b_v = b_v.view(nhead, head_dim, -1)  # [heads, d/head, 1]

    q_reshape = mh_q.view(B, nhead, Nt, head_dim)  # [B, heads, Nt, d/head]

    # Project the multihead query into the p space from the e (d/head) space
    projected_u = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_w_k)
    projected_b = torch.einsum("bhme, hep -> bhmp", q_reshape, mh_b_k)

    # Simple forward pass through the layer with nvib
    latent_dict = nvib_layer(encoder_output, src_key_padding_mask)

    # Simple forward pass through denoising attention
    d_out_train, d_attn_train = denoising_attention_train(
        mh_q,
        d_mh_k,
        d_mh_v,
        latent_dict["pi"],
        latent_dict["z"][0],
        attn_mask=d_attn_mask,
        dropout_p=dropout,
    )

    out, attn = my_scaled_dot_product_attention(
        mh_q, mh_k, mh_v, attn_mask=attn_mask, dropout_p=dropout
    )

    d_out_eval, d_attn_eval = denoising_attention_eval(
        projected_u,
        projected_b,
        latent_dict["mu"],
        latent_dict["logvar"],
        latent_dict["pi"],
        mh_w_v,
        mh_b_v,
        latent_dict["memory_key_padding_mask"].repeat(nhead, 1).unsqueeze(1),  # [B*H, 1, Nl]
        attn_mask=d_attn_mask,
        dropout_p=dropout,
    )

    # Check if output is the same
    assert torch.allclose(d_out_train, out)
    assert torch.allclose(d_out_eval, out)

    # Check if attention is the same
    assert torch.allclose(d_attn_train[:, :, 1:], attn)
    assert torch.allclose(d_attn_eval[:, :, 1:], attn)

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d, dim_feedforward=4 * d, nhead=nhead, dropout=dropout
    )

    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
    transformer_decoder_copy = nn.TransformerDecoder(decoder_layer, num_layers=1)
    # transformer_decoder_nvib = nn.TransformerDecoder(decoder_layer, num_layers=1)

    # transformer_decoder_nvib.eval()
    transformer_decoder.eval()
    transformer_decoder_copy.eval()
    for layer_num, layer in enumerate(transformer_decoder_copy.layers):
        layer.multihead_attn = torch.nn.modules.MultiheadAttention(embed_dim=d, num_heads=1)
    # Replace MultiheadAttention with DenoisingMultiheadAttention
    # for layer_num, layer in enumerate(transformer_decoder_nvib.layers):
    #     layer.multihead_attn = MultiheadAttention2(
    #         embed_dim=d, num_heads=nhead, dropout=dropout, bias=False
    #     )

    # Run Transformer decoder
    # Check input equivalence
    assert torch.allclose(encoder_output, latent_dict["z"][0][1:, :, :])
    assert torch.allclose(src_key_padding_mask, latent_dict["memory_key_padding_mask"][:, 1:])

    seed_everything(42)
    mha_out, mha_weights = transformer_decoder.layers[0].multihead_attn.forward(
        tgt, encoder_output, encoder_output, src_key_padding_mask
    )
    # mha_outCOPY, mha_weightsCOPY = transformer_decoder.layers[0].multihead_attn.forward(
    #     tgt, encoder_output, encoder_output, src_key_padding_mask
    # )
    # Yes our inputs are good!
    seed_everything(42)
    mha_outCOPY, mha_weightsCOPY = transformer_decoder_copy.layers[0].multihead_attn.forward(
        tgt,
        encoder_output,
        encoder_output,
        src_key_padding_mask,
    )
    # (z, mu, logvar, pi) = latent_dict["z"]
    # latent_dict["z"] = (z[1:, :, :], mu, logvar, pi)
    # latent_dict["memory_key_padding_mask"] = latent_dict["memory_key_padding_mask"][:, 1:]
    # mha_outNVIB, mha_weightsNVIB = transformer_decoder_nvib.layers[0].multihead_attn.forward(
    #     tgt,
    #     latent_dict["z"][0][1:, :, :],
    #     latent_dict["z"][0][1:, :, :],
    #     latent_dict["memory_key_padding_mask"][:, 1:],
    # )
    # The copy is the same!
    breakpoint()
    assert torch.allclose(mha_out, mha_outCOPY)
    assert torch.allclose(mha_weights, mha_weightsCOPY)

    # assert torch.allclose(mha_out, mha_outNVIB)
    # assert torch.allclose(mha_weights, mha_weightsNVIB)

    # print(output_nvib[:, 0, :10])
    print(output[:, 0, :10])

    seed_everything(42)
    breakpoint()
    output = transformer_decoder(
        tgt=tgt,
        memory=encoder_output,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=src_key_padding_mask,
    )

    breakpoint()
    seed_everything(42)
    # output_nvib = transformer_decoder_nvib(
    #     tgt=tgt,
    #     memory=latent_dict["z"],
    #     tgt_key_padding_mask=tgt_key_padding_mask,
    #     memory_key_padding_mask=latent_dict["memory_key_padding_mask"],
    # )

    breakpoint()
    print(output_nvib[:, 0, :10])
    print(output[:, 0, :10])
    assert torch.allclose(output, output_nvib)

    # Using profiler to analyze execution time

    # Using profiler to analyze memory consumption

    # Using tracing functionality

    # Examining stack traces

    # Visualizing data as a flame graph

    # Using profiler to analyze long-running jobs


if __name__ == "__main__":
    main()
