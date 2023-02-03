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

# Note:
# Ns: Source length
# Nt: target length
# Nl: latent length
# B: batch size
# H: hidden dimension


def inverse_cdf_approximation(alpha):
    """
    Takes in alpha values and uses them to calculate the INVCDF approximation to the Gamma
    GammaDist = 1/b * (u * a * gammaFunc(a)) ** (1/a)
    Took logs in the calculation to prevent overflow!
    b = beta = 1

    Nota Bene: This approximation is only good when alphas are small (roughly less than 1)
    :param alpha: positive values
    :return: gamma variables
    """

    u_obj = torch.distributions.uniform.Uniform(0 + 1e-8, 1 - 1e-8)
    u = u_obj.sample(alpha.size()).to(alpha.device)
    gammas = torch.exp((1 / alpha) * (torch.log(u) + torch.log(alpha) + torch.lgamma(alpha)))

    return gammas


def gauss_approximation(alpha):
    """
    Takes in alpha values and uses them to calculate the gaussian reparameterisation approximation to the Gamma
    G ~ N(a, sqrt(a))
    a = mean
    a^2 = variance

     Nota Bene: This approximation is only good when alphas are large (roughly greater than 10)
    :param alpha: positive values
    :return: gamma variables
    """

    # Training the alphas have noise
    std = torch.sqrt(alpha)
    eps = torch.randn_like(std)
    gammas = eps.mul(std).add_(alpha)
    gammas = torch.clamp(gammas, min=1e-8)  # Make sure they are not negative

    return gammas


class Nvib(nn.Module):
    """
    A Nonparameteric variational information bottleneck layer
    """

    def __init__(self, size_in, size_out, prior_mu, prior_var, prior_alpha, delta, kappa):
        super().__init__()

        # Priors
        self.prior_mu = float(prior_mu)
        self.prior_var = float(prior_var)
        self.prior_alpha = float(prior_alpha)
        self.delta = float(delta)  # Conditional prior delta
        self.kappa = int(kappa)  # Number of samples

        # Layers for parameters
        self.size_in = size_in
        self.size_out = size_out
        self.relu_activation = nn.ReLU()  # Relu for alphas
        self.mu_proj = nn.Linear(size_in, size_out)  # Project to mean
        self.logvar_proj = nn.Linear(size_in, size_out)  # Project log variance
        self.alpha_proj = nn.Linear(size_in, 1)  # Project to model size

    def reparameterize_gaussian(self, mu, logvar):

        """
        Reparameterise for gaussian
        Train = sample
        Test = mean

        :param mu: means [Nl,B,H]
        :param logvar: logged variances [Nl,B,H]
        :return: z: sample from a gaussian distribution or mean
        """

        if self.training:
            std = torch.exp(0.5 * logvar)  # [Nl,B,H]
            eps = torch.randn_like(std)  # [Nl,B,H]
            z = eps.mul(std).add_(mu)  # [Nl,B,H]
        else:
            z = mu  # [Nl,B,H]
        return z  # [Nl,B,H]

    def reparameterize_dirichlet(self, alpha, memory_key_padding_mask):

        """
        Takes in alpha parameters and returns pi from a dirichlet distribution using approximation:
        Inverse CDF approximation
        Gaussian approxiation

        :param alpha: [Nl,B,1]
        :param memory_key_padding_mask: Mask for the latent space [B,Nl]
        :return: pi [Nl,B,1]
        """

        # Get mask
        alpha_mask = memory_key_padding_mask + alpha.le(0)
        alpha = torch.clamp(alpha.clone(), min=1e-8)
        appoximation_mask = torch.gt(alpha, 0.63).float()

        if self.training:
            # Gaussian approximation or inverse cdf approximation
            gammas = appoximation_mask * gauss_approximation(alpha) + (
                1 - appoximation_mask
            ) * inverse_cdf_approximation(alpha)

        # Testing the alphas don't have noise
        else:
            gammas = alpha

        # mask and normalise (make sure its non-zero)
        gammas.masked_fill_(alpha_mask, 0)
        normalising_sum = torch.sum(gammas, 0).unsqueeze(0)
        pi = torch.div(gammas, normalising_sum)

        return pi

    def sample(self, number_samples, memory_key_padding_mask, device, *args, **kwargs):
        """
         Take a sample from the prior distribution and decode it.

         Sampling is done when the model is in evaluation mode (no dropout).
         There is an equivalence between the training and evaluation time attention functions if:
         mu = Z and variance = 0 we get the same function.

         Sample a uniform distribution of the min_length max_length and
        :param number_samples: This is like batch size
        :param memory_key_padding_mask: This is a mask that determines the lengths used [B, Nl]
        :param device:
        :param args:
        :param kwargs:
        :return: z: (z, pi, z, logvar) tuple for the decoder that uses denoising attention

        """

        # Sample from a gaussian
        memory_key_padding_mask = memory_key_padding_mask.repeat(1, self.kappa)
        max_length = memory_key_padding_mask.size(-1)
        eps = torch.randn(
            size=(max_length, number_samples, self.size_out), device=device
        )  # [Ns,B,H]
        z = self.prior_mu + (self.prior_var**0.5) * eps
        z.masked_fill_(memory_key_padding_mask.T.unsqueeze(-1), 0)
        logvar = torch.ones_like(z) * -200  # When exponentiated it will be 0

        # Sample from Dir((alpha1 + K0 * delta)/K0, ... )
        # When delta = 0 (Dirichlet process) Dir((alpha0/K0, ... ,alpha0/K0)
        # When delta = 1 (Full conditional prior) Dir((alpha0, ... ,alpha0)
        K0 = torch.sum(~memory_key_padding_mask.T.unsqueeze(-1), 0)
        alphas = (
            torch.ones(size=(max_length, number_samples, 1), device=device)
            * (self.prior_alpha + (self.delta * (K0 - 1)))
            / K0
        )
        alphas.masked_fill_(memory_key_padding_mask.T.unsqueeze(-1), 0)
        pi = self.reparameterize_dirichlet(alphas, memory_key_padding_mask.T.unsqueeze(-1))

        # This is how the decoder gets the parameters
        z_tuple = (z, pi, z, logvar)

        return z_tuple, memory_key_padding_mask

    def forward(self, encoder_output, src_key_padding_mask):
        """
        The latent layer for NVIB. Notice length comes in as NS and exits Nl (Ns+1) for the prior
        :param encoder_output:[Ns,B,H]
        :param src_key_padding_mask: [B,Ns]
        :return: A dictionary of outputs:
                z: reparameterised from the latent layer [Nl,B,H]
                pi: probability [Nl,B,1]
                memory_key_padding_mask: from the latent layer [B,Nl]
                mu: means from the latent layer [Nl,B,H]
                logvar: logged variances from the latent layer [Nl,B,H]
                alpha: psuedo-counts from the latent layer [Nl,B,H]


        """

        # Project to mean, log variance and log alpha
        mu = self.mu_proj(encoder_output)
        logvar = self.logvar_proj(encoder_output)
        alpha = self.relu_activation(self.alpha_proj(encoder_output))

        # Unknowns
        unknown_mu = torch.ones_like(mu, device=mu.device)[0, :, :].unsqueeze(0) * self.prior_mu
        unknown_logvar = torch.ones_like(logvar, device=logvar.device)[0, :, :].unsqueeze(
            0
        ) * math.log(self.prior_var)
        unknown_alpha = (
            torch.ones_like(alpha, device=alpha.device)[0, :, :].unsqueeze(0) * self.prior_alpha
        )
        mu = torch.cat((unknown_mu, mu), 0)
        logvar = torch.cat((unknown_logvar, logvar), 0)
        alpha = torch.cat((unknown_alpha, alpha), 0)

        # Include mask for unknowns
        mask = src_key_padding_mask.T.unsqueeze(-1)
        unknown_mask = torch.zeros_like(mask, dtype=bool, device=mask.device)[0, :, :].unsqueeze(0)
        mask = torch.cat((unknown_mask, mask), 0)

        # Multi sample
        if self.training:

            Nl, B, H = mu.shape  # [Nl,B,1]

            # Reparameterise component
            rho = self.reparameterize_dirichlet(alpha, mask)
            rho = rho.view(1, Nl, B, 1).repeat(self.kappa, 1, 1, 1)  # [kappa,Nl,B,1]

            # Repeat for multisampling
            mu = mu.view(1, Nl, B, H).repeat(self.kappa, 1, 1, 1)  # [kappa,Nl,B,H]
            logvar = logvar.view(1, Nl, B, H).repeat(self.kappa, 1, 1, 1)  # [kappa,Nl,B,H]
            mask = mask.view(1, Nl, B, 1).repeat(self.kappa, 1, 1, 1)  # [kappa * Nl,B,1]
            alpha = alpha.view(1, Nl, B, 1).repeat(self.kappa, 1, 1, 1) / self.kappa

            # Reparameterise
            z = self.reparameterize_gaussian(mu, logvar).view(Nl * self.kappa, B, H)
            sub_rho = self.reparameterize_dirichlet(alpha, mask)

            # Combine multisample
            pi = (rho * sub_rho).view(Nl * self.kappa, B, 1)  # [Nl,B,1]

            # Reshape
            mask = mask.view(Nl * self.kappa, B, 1)  # [kappa*Nl,B,1]
            mu = mu.view(Nl * self.kappa, B, H)  # [kappa*Nl,B,H]
            logvar = logvar.view(Nl * self.kappa, B, H)  # [kappa*Nl,B,H]
            alpha = alpha.view(Nl * self.kappa, B, 1)  # [kappa*Nl,B,1]

        else:
            # Reparameterise
            z = self.reparameterize_gaussian(mu, logvar)
            pi = self.reparameterize_dirichlet(alpha, mask)

        # mask the parameters
        alpha.masked_fill_(mask, 0)
        z.masked_fill_(mask, 0)
        mu.masked_fill_(mask, 0)
        logvar.masked_fill_(mask, 0)

        # Transform back [B,Ns]
        memory_key_padding_mask = mask.T.squeeze(0)

        # Logging
        avg_num_vec = torch.mean(torch.count_nonzero(alpha, dim=0).float())
        avg_prop_vec = torch.mean(torch.count_nonzero(alpha, dim=0) / torch.sum(~mask, 0))
        avg_alpha0 = torch.mean(torch.sum(alpha, 0))

        # This is how the decoder gets the parameters
        z_tuple = (z, pi, mu, logvar)

        return {
            "z": z_tuple,
            "pi": pi,
            "memory_key_padding_mask": memory_key_padding_mask,
            "mu": mu,
            "logvar": logvar,
            "alpha": alpha,
            "avg_num_vec": float(avg_num_vec),
            "avg_prop_vec": float(avg_prop_vec),
            "avg_alpha0": float(avg_alpha0),
        }
