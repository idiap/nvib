#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
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


def eye_scaled_(tensor, scale=1.0):
    with torch.no_grad():
        torch.eye(*tensor.shape, out=tensor, requires_grad=tensor.requires_grad).mul_(scale)
    return tensor


def init_vector_(tensor, init_vector):
    with torch.no_grad():
        tensor.copy_(init_vector)
    return tensor


class Quadratic(torch.nn.Module):
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
        prior_log_alpha=None,
        prior_log_alpha_stdev=None,
        delta=1,
        kappa=1,
        nheads=1,
        alpha_tau=None,
        stdev_tau=None,
        mu_tau=None,
    ):
        super().__init__()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Priors
        self.prior_mu = (prior_mu if prior_mu is not None else torch.zeros(size_in)).to(
            self.device
        )  # [H]
        self.prior_var = (prior_var if prior_var is not None else torch.ones(size_in)).to(
            self.device
        )  # [H]
        self.prior_log_alpha = (
            prior_log_alpha if prior_log_alpha is not None else torch.zeros(1)
        ).to(
            self.device
        )  # [1]
        self.prior_log_alpha_stdev = (
            prior_log_alpha_stdev if prior_log_alpha_stdev is not None else torch.ones(1)
        ).to(
            self.device
        )  # [1]
        self.delta = float(delta)  # Conditional prior delta
        self.kappa = int(kappa)  # Number of samples

        # Layers for parameters
        self.size_in = size_in
        self.size_out = size_out
        self.d = int(size_in / nheads)  # dimension of the head
        self.alpha_activation = Exponential()  # projection for alphas
        self.mu_proj = nn.Linear(size_in, size_out)  # Project to mean
        self.logvar_proj = nn.Linear(size_in, size_out)  # Project log variance
        self.alpha_proj = Quadratic(size_in, 1)  # Project to model size
        self.nheads = nheads  # number of heads

        # Initialisation parameters - 0 is the prior 1 is the posterior
        self.alpha_tau = alpha_tau if alpha_tau is not None else 1
        self.stdev_tau = stdev_tau if stdev_tau is not None else 1
        self.mu_tau = mu_tau if mu_tau is not None else 1
        self.init_parameters()

    def init_parameters(self):
        """
        Initialise parameters
        """
        # Initialise mu projection
        eye_scaled_(self.mu_proj.weight, self.mu_tau)
        init_vector_(self.mu_proj.bias, self.prior_mu * (1 - self.mu_tau))

        # Initialise logvar projection
        nn.init.constant_(self.logvar_proj.weight, 0)
        init_vector_(
            self.logvar_proj.bias,
            torch.log(
                (torch.sqrt(self.prior_var) * self.stdev_tau)
                ** 2  # Controls the standard deviation
                + torch.finfo(self.prior_var.dtype).tiny
            ),  # nonzero
        )

        # Initialise alpha projection
        nn.init.constant_(self.alpha_proj.weight_quadratic, 1 / (2 * math.sqrt(self.d)))
        nn.init.constant_(self.alpha_proj.weight_linear, 0)
        init_vector_(
            self.alpha_proj.bias,
            self.prior_log_alpha_stdev * (self.alpha_tau),  # Standard deviation of log alpha
        )

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

    def reparameterize_dirichlet(self, alpha, mask):
        """
        Takes in alpha parameters and returns pi from a dirichlet distribution.

        :param alpha: [Nl,B,1]
        :param mask: Mask for the latent space [B,Nl]
        :return: pi [Nl,B,1]
        """

        if self.training:
            # Implicit gradients for Gamma (batch_shape [Nl, B]) each individual gamma
            gamma_dist = torch.distributions.Gamma(alpha, torch.ones_like(alpha))
            gammas = gamma_dist.rsample()

        # Testing the alphas don't have noise
        else:
            thresh = nn.Threshold(0.1, 0)
            gammas = thresh(alpha)

        # mask and normalise (make sure its non-zero)
        gammas.masked_fill_(mask, 0)
        normalising_sum = torch.sum(gammas, 0).unsqueeze(0) + torch.finfo(gammas.dtype).tiny
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
        pi = self.reparameterize_dirichlet(alphas, memory_key_padding_mask.T.unsqueeze(-1))

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

        alpha = alpha.masked_fill(memory_key_padding_mask.transpose(1, 0).unsqueeze(-1), 0)
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
        alpha = alpha.masked_fill(memory_key_padding_mask.transpose(1, 0).unsqueeze(-1), 0)
        alpha0_q = torch.sum(alpha, 0).squeeze(-1).to(torch.float64)  # [B]
        alpha0_p = (torch.ones_like(alpha0_q) * (self.prior_alpha + lowerBound)).to(
            torch.float64
        )  # [B]

        kl = (
            torch.lgamma(alpha0_q)
            - torch.lgamma(alpha0_p)
            + (alpha0_q - alpha0_p) * (-torch.digamma(alpha0_q) + torch.digamma(alpha0_q / k0))
            + k0 * (torch.lgamma(alpha0_p / k0) - torch.lgamma(alpha0_q / k0))
        ) / n

        return kl.to(torch.float32)

    def forward(
        self, encoder_output, mask, alpha_skip=None, batch_first=True, logging=False, **kwargs
    ):
        """
        The latent layer for NVIB. Notice length comes in as NS and exits Nl (Ns+1) for the prior
        :param encoder_output:[B, Ns, P]
        :param mask: [B,Ns] Always batch first
        :param alpha_skip: [B,Nl,heads]
        :param batch_first: True if batch first
        :param include_prior_component: True if including prior component - remove in autoregressive decoding
        :param logging: True if logging metrics
        :return: A dictionary of outputs:
                z: (z, pi, mu, logvar) tuple where inner z is reparameterised from the latent layer [B, Nl, P]
                pi: probability [B,Nl,1]
                memory_key_padding_mask: from the latent layer [B,Nl]
                mu: means from the latent layer [B,Nl,P]
                logvar: logged variances from the latent layer [B, Nl, P]
                alpha: psuedo-counts from the latent layer [B,Nl,heads]


        """
        # If batch first, transpose.
        if batch_first:
            encoder_output = encoder_output.transpose(1, 0)
            if alpha_skip is not None:
                alpha_skip = alpha_skip.transpose(1, 0)

        # Useful dimensions
        Ns, B, H = encoder_output.shape
        Nl = Ns + 1

        # Project to mean, log variance and log alpha
        mu = self.mu_proj(encoder_output)
        logvar = self.logvar_proj(encoder_output)

        # Alpha skip connection in log space
        if alpha_skip is not None:
            log_alpha = self.alpha_proj(encoder_output) + torch.log(alpha_skip[1:, :, :])
            alpha = self.alpha_activation(log_alpha)
        else:
            log_alpha = self.alpha_proj(encoder_output)
            alpha = self.alpha_activation(log_alpha)

        # Clamp alpha
        # alpha = torch.clamp(alpha, min=0, max=torch.finfo(alpha.dtype).max - 1000)
        mask = mask.transpose(1, 0).unsqueeze(-1)
        # Unknowns are the prior [1, B, P]

        unknown_mu = torch.ones_like(mu[0:1, :, :], device=self.device) * self.prior_mu
        unknown_logvar = torch.ones_like(logvar[0:1, :, :], device=self.device) * torch.log(
            self.prior_var
        )
        # Size is [1, B, 1]
        unknown_log_alpha = (
            torch.ones_like(alpha[0:1, :, :], device=self.device) * self.prior_log_alpha
        )

        mu = torch.cat((unknown_mu, mu), 0)
        logvar = torch.cat((unknown_logvar, logvar), 0)
        alpha = torch.cat((self.alpha_activation(unknown_log_alpha), alpha), 0)
        log_alpha = torch.cat((unknown_log_alpha, log_alpha), 0)

        # Include mask for unknowns [Nl,B,1]
        unknown_mask = torch.zeros_like(mask[0:1, :, :], dtype=bool, device=self.device)
        mask = torch.cat((unknown_mask, mask), 0)

        # Multi sample
        if self.kappa > 1:
            # Reparameterise component
            rho = self.reparameterize_dirichlet(alpha, mask)
            rho = rho.view(1, Nl, B, 1).repeat(self.kappa, 1, 1, 1)  # [kappa,Nl,B,1]

            # Repeat for multisampling
            mu = mu.view(1, Nl, B, H).repeat(self.kappa, 1, 1, 1)  # [kappa,Nl,B,P]
            logvar = logvar.view(1, Nl, B, H).repeat(self.kappa, 1, 1, 1)  # [kappa,Nl,B,P]
            mask = mask.view(1, Nl, B, 1).repeat(self.kappa, 1, 1, 1)  # [kappa * Nl,B,1]
            alpha = alpha.view(1, Nl, B, 1).repeat(self.kappa, 1, 1, 1) / self.kappa

            # Reparameterise
            z = self.reparameterize_gaussian(mu, logvar).view(Nl * self.kappa, B, H)
            sub_rho = self.reparameterize_dirichlet(alpha, mask)

            # Combine multisample
            pi = (rho * sub_rho).view(Nl * self.kappa, B, 1)  # [Nl,B,1]

            # Reshape
            mask = mask.view(Nl * self.kappa, B, 1)  # [kappa*Nl,B,1]
            mu = mu.view(Nl * self.kappa, B, H)  # [kappa*Nl,B,P]
            logvar = logvar.view(Nl * self.kappa, B, H)  # [kappa*Nl,B,P]
            alpha = alpha.view(Nl * self.kappa, B, 1)  # [kappa*Nl,B,1]

        else:
            # Reparameterise
            z = self.reparameterize_gaussian(mu, logvar)
            pi = self.reparameterize_dirichlet(alpha, mask)

        # Reshape for batch first
        if batch_first:
            z = z.transpose(1, 0)
            pi = pi.transpose(1, 0)
            mu = mu.transpose(1, 0)
            logvar = logvar.transpose(1, 0)
            alpha = alpha.transpose(1, 0)
            log_alpha = log_alpha.transpose(1, 0)

        # Logging
        if logging:
            avg_num_vec = torch.mean(torch.count_nonzero(pi.masked_fill(mask, 0), dim=0).float())
            avg_prop_vec = torch.mean(
                torch.count_nonzero(pi.masked_fill(mask, 0), dim=0) / torch.sum(~mask, 0)
            )
            avg_alpha0 = torch.mean(torch.sum(alpha.masked_fill(mask, 0), 0))

            return {
                "z": (z, pi, mu, logvar),  # This is how the decoder gets the parameters
                "pi": pi,
                "memory_key_padding_mask": mask.transpose(2, 0).squeeze(0),  # [B,Nl]
                "mu": mu,
                "logvar": logvar,
                "alpha": alpha,
                "log_alpha": log_alpha,
                "avg_num_vec": float(avg_num_vec),
                "avg_prop_vec": float(avg_prop_vec),
                "avg_alpha0": float(avg_alpha0),
            }
        # No logging
        else:
            return {
                "z": (z, pi, mu, logvar),  # This is how the decoder gets the parameters
                "pi": pi,
                "memory_key_padding_mask": mask.transpose(2, 0).squeeze(0),  # [B,Nl]
                "mu": mu,
                "logvar": logvar,
                "alpha": alpha,
                "log_alpha": log_alpha,
            }
