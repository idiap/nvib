#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch


def kl_gaussian(
    mu, logvar, alpha, memory_key_padding_mask, prior_mu=0, prior_var=1, kappa=1, **kwargs
):
    """
    KL Loss for the Gaussian component with expected K
    :param mu: mean [Nl,B,H]
    :param logvar: logged variance [Nl,B,H]
    :param alpha: psuedo count weight [Nl,B,1]
    :param memory_key_padding_mask: boolean mask [B,Nl]
    :param prior_mu: prior for the mean [1]
    :param prior_var: prior for the variance [1]
    :return: KL [B]
    """

    # Scaling
    # Total number of vectors sampled
    k0 = torch.sum(~memory_key_padding_mask.T, 0)  # [B]
    # Input length
    n = k0 / kappa  # [B]

    alpha0_q = torch.sum(alpha.T, -1)  # [1,B]
    expected_pi = alpha.squeeze(-1) / alpha0_q  # [Nl,B]

    # KL between univariate Gaussians
    var_ratio = logvar.exp() / prior_var
    t1 = (mu - prior_mu) ** 2 / prior_var
    kl = var_ratio + t1 - 1 - var_ratio.log()
    kl = kl.masked_fill(memory_key_padding_mask.T.unsqueeze(-1), 0)

    # Mean over embedding dimension
    kl = torch.mean(kl, -1)  # [Nl,B]

    # Scale and sum over sentence length dimension
    kl = 0.5 * k0 * torch.sum(kl * expected_pi, 0)
    kl /= n

    return kl


def kl_dirichlet(alpha, memory_key_padding_mask, prior_alpha=1, delta=1, kappa=1, **kwargs):
    """
    The regularisation for the dirichlet component with expected K

    :param alpha: k dimensional psuedo counts [Nl,B,1]
    :param memory_key_padding_mask: boolean mask [B,Nl]
    :param prior_alpha: prior for the psuedo-counts [1]
    :param delta: the control on the conditional prior [1]
    :param kappa: the number of samples [1]
    :return: Kl [B]

    Nota Bene: digamma and lgamma cannot be zero
    """

    # Total number of vectors sampled
    k0 = torch.sum(~memory_key_padding_mask.T, 0)  # [B]
    # Input length
    n = k0 / kappa  # [B]
    # Conditional prior lower bound. Sentence length without prior
    lowerBound = delta * (n - 1)

    # Sum the alphas (already masked)
    alpha0_q = torch.sum(alpha, 0).squeeze(-1)  # [B]
    alpha0_p = torch.ones_like(alpha0_q) * (prior_alpha + lowerBound)  # [B]

    kl = torch.lgamma(alpha0_q) - torch.lgamma(alpha0_p)
    kl += (alpha0_q - alpha0_p) * (-torch.digamma(alpha0_q) + torch.digamma(alpha0_q / k0))
    kl += k0 * (torch.lgamma(alpha0_p / k0) - torch.lgamma(alpha0_q / k0))
    kl /= n
    return kl
