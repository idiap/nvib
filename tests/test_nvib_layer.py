#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import math

import torch
import torch.nn as nn
from torch.nn.functional import linear
from torch.nn.init import xavier_uniform_

from nvib.denoising_attention import (  # DenoisingMultiheadAttention,
    denoising_attention_eval,
    denoising_attention_train,
    pytorch_scaled_dot_product_attention,
)
from nvib.nvib_layer import Nvib

########################################################################################
# Test Nvib layer
########################################################################################

torch.manual_seed(42)

# Set parameter sizes
Ns, Nt, B, P, d = 10, 6, 2, 32, 32
nhead = 8
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
latent_dict = nvib_layer(encoder_output, src_key_padding_mask, batch_first=False)


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

# # Define Transformer decoder
# decoder_layer = nn.TransformerDecoderLayer(
#     d_model=d, dim_feedforward=4 * d, nhead=nhead, dropout=dropout
# )
# transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

# # Replace MultiheadAttention with DenoisingMultiheadAttention
# for layer_num, layer in enumerate(transformer_decoder.layers):
#     layer.multihead_attn = DenoisingMultiheadAttention(
#         embed_dim=d, num_heads=nhead, dropout=dropout, bias=True
#     )

# # Run Transformer decoder
# output = transformer_decoder(
#     tgt=tgt,
#     memory=latent_dict["z"],
#     tgt_key_padding_mask=tgt_key_padding_mask,
#     memory_key_padding_mask=latent_dict["memory_key_padding_mask"],
# )


class Quadratic(torch.nn.Module):
    def __init__(self, size_in, size_out):
        """
        In the constructor we instantiate three parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out

        bias = torch.Tensor(size_out)
        weight_linear = torch.Tensor(size_in, size_out)
        weight_quadratic = torch.Tensor(size_in, size_out)

        self.bias = torch.nn.Parameter(bias)
        self.weight_linear = torch.nn.Parameter(weight_linear)
        self.weight_quadratic = torch.nn.Parameter(weight_quadratic)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        return (
            self.bias + torch.mm(x, self.weight_linear) + torch.mm(x**2, self.weight_quadratic)
        )


class QuadraticLinear(torch.nn.Module):
    def __init__(self, size_in, size_out):
        """
        In the constructor we instantiate three parameters and assign them as
        member parameters.
        """
        super().__init__()

        self.linear = torch.nn.Linear(size_in, size_out)
        self.quadratic = torch.nn.Linear(size_in, size_out, bias=False)

        self.bias = self.linear.bias
        self.weight_linear = self.linear.weight
        self.weight_quadratic = self.quadratic.weight

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.linear(x) + self.quadratic(x**2)


def test_quadratic_layers():
    """
    Simple test for the quadratic layer implementations
    """

    quad1 = Quadratic(3, 1)
    quad2 = QuadraticLinear(3, 1)
    nn.init.constant_(quad1.bias, 1)
    nn.init.constant_(quad1.weight_linear, 1)
    nn.init.constant_(quad1.weight_quadratic, 1)

    nn.init.constant_(quad2.bias, 1)
    nn.init.constant_(quad2.weight_linear, 1)
    nn.init.constant_(quad2.weight_quadratic, 1)

    x = torch.arange(9).view(3, 3).float()

    assert torch.allclose(quad1(x), quad2(x))


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

    out, attn = pytorch_scaled_dot_product_attention(
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

    latent_dict = nvib_layer(encoder_output, src_key_padding_mask, batch_first=False)
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

    latent_dict = nvib_layer(encoder_output, src_key_padding_mask, batch_first=False)
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

    latent_dict = nvib_layer(encoder_output, src_key_padding_mask, batch_first=False)

    sum_pi = torch.sum(latent_dict["pi"], dim=0)
    assert torch.allclose(sum_pi, torch.ones_like(sum_pi))


def main():
    breakpoint()


if __name__ == "__main__":
    main()
