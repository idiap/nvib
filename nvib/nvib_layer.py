#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch
import torch.nn as nn

# Note:
# Ns: Source length
# Nt: target length
# Nl: latent length
# B: batch size
# H: hidden dimension


class Exponential(nn.Module):
    """
    Simple exponential activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class Nvib(nn.Module):
    """
    A Nonparameteric variational information bottleneck layer
    """

    def __init__(
        self,
        size_in,
        size_out,
        prior_mu=None,
        prior_var=None,
        prior_alpha=None,
        delta=1,
        kappa=1,
        nheads=1,
    ):
        super().__init__()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Priors
        self.prior_mu = (prior_mu if prior_mu is not None else torch.zeros(size_in)).to(
            self.device
        )  # [H]
        self.prior_var = (
            prior_var if prior_var is not None else torch.ones(size_in)
        ).to(
            self.device
        )  # [H]
        self.prior_alpha = (
            prior_alpha if prior_alpha is not None else torch.ones(1)
        ).to(
            self.device
        )  # [1]
        self.delta = float(delta)  # Conditional prior delta
        self.kappa = int(kappa)  # Number of samples

        # Layers for parameters
        self.size_in = size_in
        self.size_out = size_out
        self.alpha_activation = Exponential()  # activation for alphas
        self.mu_proj = nn.Linear(size_in, size_out)  # Project to mean
        self.logvar_proj = nn.Linear(size_in, size_out)  # Project log variance
        self.alpha_proj = nn.Linear(size_in, 1)  # Project to model size
        self.d = size_in / nheads  # dimension of the head

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
        Takes in alpha parameters and returns pi from a dirichlet distribution.

        :param alpha: [Nl,B,1]
        :param memory_key_padding_mask: Mask for the latent space [B,Nl]
        :return: pi [Nl,B,1]
        """

        # Get mask
        alpha_mask = memory_key_padding_mask + alpha.le(0)
        alpha = torch.clamp(alpha.clone(), min=1e-14)

        if self.training:
            # Implicit gradients for Gamma (batch_shape [Nl, B]) each individual gamma
            gamma_dist = torch.distributions.Gamma(alpha, torch.ones_like(alpha))
            gammas = gamma_dist.rsample()

        # Testing the alphas don't have noise
        else:
            thresh = nn.Threshold(0.1, 0)
            gammas = thresh(alpha)
            alpha_mask = memory_key_padding_mask + gammas.le(0)
            # gammas = alpha

        # mask and normalise (make sure its non-zero)
        gammas.masked_fill_(alpha_mask, 0)
        normalising_sum = torch.sum(gammas, 0).unsqueeze(0) + 1e-14
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
        z.masked_fill_(memory_key_padding_mask.transpose(1, 0).unsqueeze(-1), 0)
        logvar = torch.ones_like(z) * -200  # When exponentiated it will be 0

        # Sample from Dir((alpha1 + K0 * delta)/K0, ... )
        # When delta = 0 (Dirichlet process) Dir((alpha0/K0, ... ,alpha0/K0)
        # When delta = 1 (Full conditional prior) Dir((alpha0, ... ,alpha0)
        K0 = torch.sum(~memory_key_padding_mask.transpose(1, 0).unsqueeze(-1), 0)
        alphas = (
            torch.ones(size=(max_length, number_samples, 1), device=device)
            * (self.prior_alpha + (self.delta * (K0 - 1)))
            / K0
        )
        alphas.masked_fill_(memory_key_padding_mask.transpose(1, 0).unsqueeze(-1), 0)
        pi = self.reparameterize_dirichlet(
            alphas, memory_key_padding_mask.T.unsqueeze(-1)
        )

        # This is how the decoder gets the parameters
        z_tuple = (z, pi, z, logvar)

        return z_tuple, memory_key_padding_mask

    def kl_gaussian(self, mu, logvar, alpha, memory_key_padding_mask, **kwargs):
        """
        KL Loss for the Gaussian component with expected K
        :param mu: mean [Nl,B,H]
        :param logvar: logged variance [Nl,B,H]
        :param alpha: psuedo count weight [Nl,B,1]
        :param memory_key_padding_mask: boolean mask [B,Nl]
        :return: KL [B]
        """

        # Scaling
        # Total number of vectors sampled
        k0 = torch.sum(~memory_key_padding_mask.transpose(1, 0), 0)  # [B]
        # Input length
        n = k0 / self.kappa  # [B]

        alpha = alpha.masked_fill(
            memory_key_padding_mask.transpose(1, 0).unsqueeze(-1), 0
        )
        alpha0_q = torch.sum(alpha.transpose(2, 0), -1)  # [1,B]
        expected_pi = alpha.squeeze(-1) / alpha0_q  # [Nl,B]

        # KL between univariate Gaussians
        var_ratio = logvar.exp() / self.prior_var
        t1 = (mu - self.prior_mu) ** 2 / self.prior_var
        kl = var_ratio + t1 - 1 - var_ratio.log()
        kl = kl.masked_fill(memory_key_padding_mask.transpose(1, 0).unsqueeze(-1), 0)

        # Mean over embedding dimension
        kl = torch.mean(kl, -1)  # [Nl,B]

        # Scale and sum over sentence length dimension
        kl = 0.5 * k0 * torch.sum(kl * expected_pi, 0)
        kl /= n

        return kl

    def kl_dirichlet(self, alpha, memory_key_padding_mask, **kwargs):
        """
        The regularisation for the dirichlet component with expected K

        :param alpha: k dimensional psuedo counts [Nl,B,1]
        :param memory_key_padding_mask: boolean mask [B,Nl]
        :return: Kl [B]

        Nota Bene: digamma and lgamma cannot be zero
        """

        # Total number of vectors sampled
        k0 = torch.sum(~memory_key_padding_mask.transpose(1, 0), 0)  # [B]
        # Input length
        n = k0 / self.kappa  # [B]
        # Conditional prior lower bound. Sentence length without prior
        lowerBound = self.delta * (n - 1)

        # Sum the alphas
        alpha = alpha.masked_fill(
            memory_key_padding_mask.transpose(1, 0).unsqueeze(-1), 0
        )
        alpha0_q = torch.sum(alpha, 0).squeeze(-1).to(torch.float64)  # [B]
        alpha0_p = (torch.ones_like(alpha0_q) * (self.prior_alpha + lowerBound)).to(
            torch.float64
        )  # [B]

        kl = (
            torch.lgamma(alpha0_q)
            - torch.lgamma(alpha0_p)
            + (alpha0_q - alpha0_p)
            * (-torch.digamma(alpha0_q) + torch.digamma(alpha0_q / k0))
            + k0 * (torch.lgamma(alpha0_p / k0) - torch.lgamma(alpha0_q / k0))
        ) / n

        return kl.to(torch.float32)

    def forward(self, encoder_output, src_key_padding_mask, alpha_skip=None):
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
        # Alpha skip connection in log space
        if alpha_skip is not None:
            alpha = self.alpha_activation(
                self.alpha_proj(encoder_output) + torch.log(alpha_skip[1:, :, :])
            )
        else:
            alpha = self.alpha_activation(self.alpha_proj(encoder_output))

        # Unknowns
        unknown_mu = (
            self.prior_mu.unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, mu.shape[1], 1)
            .to(self.device)
        )  # [1,B,H]
        unknown_logvar = (
            torch.log(self.prior_var)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, logvar.shape[1], 1)
        ).to(
            self.device
        )  # [1,B,H]
        unknown_alpha = (
            self.prior_alpha.unsqueeze(0).unsqueeze(0).repeat(1, alpha.shape[1], 1)
        ).to(
            self.device
        )  # [1,B,1]

        mu = torch.cat((unknown_mu, mu), 0)
        logvar = torch.cat((unknown_logvar, logvar), 0)
        alpha = torch.cat((unknown_alpha, alpha), 0)

        # Include mask for unknowns
        mask = src_key_padding_mask.transpose(1, 0).unsqueeze(-1)
        unknown_mask = torch.zeros_like(mask, dtype=bool, device=mask.device)[
            0, :, :
        ].unsqueeze(0)
        mask = torch.cat((unknown_mask, mask), 0)

        # Multi sample
        if self.training:
            Nl, B, H = mu.shape  # [Nl,B,1]

            # Reparameterise component
            rho = self.reparameterize_dirichlet(alpha, mask)
            rho = rho.view(1, Nl, B, 1).repeat(self.kappa, 1, 1, 1)  # [kappa,Nl,B,1]

            # Repeat for multisampling
            mu = mu.view(1, Nl, B, H).repeat(self.kappa, 1, 1, 1)  # [kappa,Nl,B,H]
            logvar = logvar.view(1, Nl, B, H).repeat(
                self.kappa, 1, 1, 1
            )  # [kappa,Nl,B,H]
            mask = mask.view(1, Nl, B, 1).repeat(
                self.kappa, 1, 1, 1
            )  # [kappa * Nl,B,1]
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

        # Transform back [B,Ns]
        memory_key_padding_mask = mask.transpose(2, 0).squeeze(0)

        # Logging
        avg_num_vec = torch.mean(
            torch.count_nonzero(pi.masked_fill(mask, 0), dim=0).float()
        )
        avg_prop_vec = torch.mean(
            torch.count_nonzero(pi.masked_fill(mask, 0), dim=0) / torch.sum(~mask, 0)
        )
        avg_alpha0 = torch.mean(torch.sum(alpha.masked_fill(mask, 0), 0))

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
