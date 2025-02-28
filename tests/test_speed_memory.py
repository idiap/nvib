# Set directory one higher
import os.path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import random

import torch
from torch.nn.functional import linear
from torch.nn.init import xavier_uniform_
from tqdm.auto import tqdm

from nvib.denoising_attention import (
    denoising_attention_eval,
    denoising_attention_train,
    pytorch_scaled_dot_product_attention,
)
from nvib.nvib_layer import Nvib


def generate_random_masks(batch, len):
    # returns a random boolean mask with shape (batch, len) true is where to mask

    # Generate random lengths for the sequence
    seq = torch.tensor(random.choices(list(range(1, len)), k=batch))
    ones = seq.new_ones(seq.size(0), len)
    range_tensor = ones.cumsum(dim=1)
    return seq.unsqueeze(1) < range_tensor


def random_model_initialisation(seed=42, tie_p_d=True, is_cross=True, **kwargs):
    device = (
        kwargs["device"]
        if "device" in kwargs
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # print("Using device:", device)
    atol = kwargs["atol"] if "atol" in kwargs else 1e-4

    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)

    # Set parameter sizes (we assume P = d)
    # Ns = number of source tokens
    # Nt = number of target tokens
    # b = batch size
    # p = embedding size
    # d = model dimension
    # h = number of heads
    # kappa = samples in the latent space

    Ns = kwargs["Ns"] if "Ns" in kwargs else random.randint(2, 256)
    Nt = kwargs["Nt"] if "Nt" in kwargs else random.randint(3, 256)
    b = kwargs["b"] if "b" in kwargs else random.randint(1, 128)

    # Select heads and make sure h divides d
    h = kwargs["h"] if "h" in kwargs else random.choice([1, 2, 4, 8, 16])

    if h == 1:
        if tie_p_d:
            p = d = 2 ** random.randint(1, 10)  # embedding sizes up to 1024
        else:
            p = 2 ** random.randint(1, 10)
            d = 2 ** random.randint(1, 10)
    elif h == 2:
        if tie_p_d:
            p = d = 2 ** random.randint(1, 9) * h  # embedding sizes up to 1024
        else:
            p = 2 ** random.randint(1, 9) * h
            d = 2 ** random.randint(1, 9) * h
    elif h == 4:
        if tie_p_d:
            p = d = 2 ** random.randint(1, 8) * h
        else:
            p = 2 ** random.randint(1, 8) * h
            d = 2 ** random.randint(1, 8) * h
    elif h == 8:
        if tie_p_d:
            p = d = 2 ** random.randint(1, 7) * h
        else:
            p = 2 ** random.randint(1, 7) * h
            d = 2 ** random.randint(1, 7) * h
    elif h == 16:
        if tie_p_d:
            p = d = 2 ** random.randint(1, 6) * h
        else:
            p = 2 ** random.randint(1, 6) * h
            d = 2 ** random.randint(1, 6) * h
    else:
        raise ValueError("h must be 1, 2, 4, 8 or 16")

    p = kwargs["p"] if "p" in kwargs else p
    d = kwargs["d"] if "d" in kwargs else d
    # check that d is divisible by h
    assert d % h == 0

    head_dim = int(d / h)
    kappa = kwargs["kappa"] if "kappa" in kwargs else 1

    # Define inputs and masks
    embedding = torch.rand(b, Ns, p)  # Key and value
    src_key_padding_mask = generate_random_masks(b, Ns)
    if is_cross:
        tgt = torch.rand(b, Nt, p)
        tgt_key_padding_mask = generate_random_masks(b, Ns)
    else:
        Nt = Ns
        tgt = embedding
        tgt_key_padding_mask = src_key_padding_mask
    dropout = kwargs["dropout"] if "dropout" in kwargs else 0.0

    ####### NVIV LAYER #######

    # Define Nvib layer
    nvib_layer = Nvib(
        size_in=p,
        size_out=d,
        prior_mu=None,
        prior_var=None,
        prior_alpha=None,
        delta=1,
        kappa=kappa,
        nheads=h,
        mu_tau=None,
        alpha_tau=None,
        stdev_tau=None,
    )

    # Forward pass through Nvib layer
    latent_dict = nvib_layer(embedding, src_key_padding_mask)

    # Checks
    assert torch.allclose(embedding, latent_dict["z"][0][:, 1:, :])

    # Fixed from the pretrained model
    in_proj_weight = torch.randn((3 * d, p))
    in_proj_bias = torch.randn(3 * d)
    xavier_uniform_(in_proj_weight)

    w_q, w_k, w_v = in_proj_weight.split([d, d, d])
    b_q, b_k, b_v = in_proj_bias.split([d, d, d])

    q = linear(tgt, w_q, b_q)
    k = linear(embedding, w_k, b_k)
    v = linear(embedding, w_v, b_v)

    # Denoising has an extra dimension
    d_k = linear(latent_dict["z"][0], w_k, b_k)
    d_v = linear(latent_dict["z"][0], w_v, b_v)

    # Before and after linear projection should be identical
    # WEIRD its not it needs a tolerance of 1e-6
    # torch.allclose(linear(latent_dict["z"][0][:, 1:, :], w_k, b_k),linear(latent_dict["z"][0], w_k, b_k)[:, 1:, :],atol=1e-6)
    assert torch.allclose(
        linear(latent_dict["z"][0][:, 1:, :], w_k, b_k),
        linear(latent_dict["z"][0], w_k, b_k)[:, 1:, :],
        atol=atol,
    )
    assert torch.allclose(k, d_k[:, 1:, :], atol=atol)
    assert torch.allclose(v, d_v[:, 1:, :], atol=atol)

    # Reshape the multihead query and weights
    mh_w_q = w_q.view(h, head_dim, -1)  # [heads, d/head, p]
    mh_w_k = w_k.view(h, head_dim, -1)  # [heads, d/head, p]
    mh_w_v = w_v.view(h, head_dim, -1)  # [heads, d/head, p]
    mh_b_k = b_k.view(h, head_dim, -1)  # [heads, d/head, 1]
    mh_b_v = b_v.view(h, head_dim, -1)  # [heads, d/head, 1]
    mh_b_q = b_q.view(h, head_dim, -1)  # [heads, d/head, p]

    # Multihead

    # Make sure the mulithead is correct with B1H1, ... ,B1Hn then B2H1, ... ,B2Hn
    # breakpoint()
    # k1 = k.permute(1, 0, 2).contiguous().view(Ns, b * h, head_dim).permute(1, 0, 2)
    # assert torch.allclose(k[0, :10, 0], k1[0, :10, 0])
    # assert not torch.allclose(k[1, :10, 0], k1[1, :10, 0])
    # assert torch.allclose(k[1, :10, 0], k1[h, :10, 0])

    mh_q = q.permute(1, 0, 2).contiguous().view(Nt, b * h, head_dim).permute(1, 0, 2)
    mh_k = k.permute(1, 0, 2).contiguous().view(Ns, b * h, head_dim).permute(1, 0, 2)
    mh_v = v.permute(1, 0, 2).contiguous().view(Ns, b * h, head_dim).permute(1, 0, 2)

    # # Multihead (denoising)
    d_mh_v = (
        d_v.permute(1, 0, 2).contiguous().view((Ns + 1) * kappa, b * h, head_dim).permute(1, 0, 2)
    )
    d_mh_k = (
        d_k.permute(1, 0, 2).contiguous().view((Ns + 1) * kappa, b * h, head_dim).permute(1, 0, 2)
    )

    assert torch.allclose(mh_v, d_mh_v[:, 1:, :], atol=atol)
    assert torch.allclose(mh_k, d_mh_k[:, 1:, :], atol=atol)

    return {
        "seed": seed,
        "atol": atol,
        "Ns": Ns,
        "Nt": Nt,
        "b": b,
        "p": p,
        "d": d,
        "h": h,
        "dropout": dropout,
        "head_dim": head_dim,
        "kappa": kappa,
        "embedding": embedding,
        "src_key_padding_mask": src_key_padding_mask,
        "tgt": tgt,
        "tgt_key_padding_mask": tgt_key_padding_mask,
        "q": mh_q,
        "k": mh_k,
        "v": mh_v,
        "d_k": d_mh_k,
        "d_v": d_mh_v,
        "latent_dict": latent_dict,
        "weights": {"w_q": mh_w_q, "w_k": mh_w_k, "w_v": mh_w_v},
        "bias": {"b_q": mh_b_q, "b_k": mh_b_k, "b_v": mh_b_v},
    }


# Test equivalence between training time attention and original attention
#   with multiple random seeds, lengths and batch sizes, heads etc.

# Test equivalence between eval time attention and original attention
#   with multiple random seeds, lengths and batch sizes, heads etc.

# Test cross attention case

# Test self attention case

# Test causal self attention case


# Training function
def test_cross_attention_train(seed=42, **kwargs):
    args = random_model_initialisation(seed, is_cross=True, **kwargs)

    # m = args["src_key_padding_mask"]
    # m1 = m.unsqueeze(1).repeat(args["h"], 1, 1) # THIS IS WRONG
    # m2 = args["src_key_padding_mask"].unsqueeze(1).repeat_interleave(args["h"], dim=0)
    # assert torch.allclose(m[0,:], m1[0,0,:])
    # assert torch.allclose(m[0,:], m2[0,0,:])
    # assert not torch.allclose(m[1,:], m1[1,0,:])
    # assert not torch.allclose(m[1,:], m2[1,0,:])

    # Standard attention
    standard_attn, standard_weights = pytorch_scaled_dot_product_attention(
        query=args["q"],
        key=args["k"],
        value=args["v"],
        attn_mask=args["src_key_padding_mask"].unsqueeze(1).repeat_interleave(args["h"], dim=0),
        dropout_p=args["dropout"],
        is_causal=False,
    )

    # Denoising attention (training)
    denoising_attn, denoising_weights = denoising_attention_train(
        query=args["q"],
        key=args["d_k"],
        value=args["d_v"],
        Z=args["latent_dict"]["z"][0],
        pi=args["latent_dict"]["pi"],
        attn_mask=args["latent_dict"]["memory_key_padding_mask"]
        .unsqueeze(1)
        .repeat_interleave(args["h"], dim=0),
        dropout_p=args["dropout"],
    )

    # Check equivalence
    assert torch.allclose(standard_attn, denoising_attn, atol=args["atol"])
    assert torch.allclose(standard_weights, denoising_weights[:, :, 1:], atol=args["atol"])


# Evaluation function
def test_cross_attention_eval(seed=42, **kwargs):
    args = random_model_initialisation(seed, is_cross=True, **kwargs)

    # m = args["src_key_padding_mask"]
    # m1 = m.unsqueeze(1).repeat(args["h"], 1, 1) # THIS IS WRONG
    # m2 = args["src_key_padding_mask"].unsqueeze(1).repeat_interleave(args["h"], dim=0)
    # assert torch.allclose(m[0,:], m1[0,0,:])
    # assert torch.allclose(m[0,:], m2[0,0,:])
    # assert not torch.allclose(m[1,:], m1[1,0,:])
    # assert not torch.allclose(m[1,:], m2[1,0,:])

    # Standard attention
    standard_attn, standard_weights = pytorch_scaled_dot_product_attention(
        query=args["q"],
        key=args["k"],
        value=args["v"],
        attn_mask=args["src_key_padding_mask"].unsqueeze(1).repeat_interleave(args["h"], dim=0),
        dropout_p=args["dropout"],
        is_causal=False,
    )

    q_reshape = args["q"].view(
        args["b"], args["h"], args["Nt"], args["head_dim"]
    )  # [B, heads, Nt, d/head]

    # Project the multihead query into the p space from the e (d/head) space
    projected_u = torch.einsum("bhme, hep -> bhmp", q_reshape, args["weights"]["w_k"])
    projected_b = torch.einsum("bhme, hep -> bhmp", q_reshape, args["bias"]["b_k"])

    # Denoising attention (training)
    denoising_attn, denoising_weights = denoising_attention_eval(
        projected_u=projected_u,
        projected_b=projected_b,
        mu=args["latent_dict"]["mu"],
        logvar=args["latent_dict"]["logvar"],
        pi=args["latent_dict"]["pi"],
        w_v=args["weights"]["w_v"],
        b_v=args["bias"]["b_v"],
        attn_mask=args["latent_dict"]["memory_key_padding_mask"]
        .unsqueeze(1)
        .repeat_interleave(args["h"], dim=0),
        dropout_p=args["dropout"],
    )

    # Check equivalence
    assert torch.allclose(standard_attn, denoising_attn, atol=args["atol"])
    assert torch.allclose(standard_weights, denoising_weights[:, :, 1:], atol=args["atol"])


# Training function
def test_self_attention_train(seed=42, **kwargs):
    args = random_model_initialisation(seed, is_cross=False, **kwargs)

    # m = args["src_key_padding_mask"]
    # m1 = m.unsqueeze(1).repeat(args["h"], 1, 1) # THIS IS WRONG
    # m2 = args["src_key_padding_mask"].unsqueeze(1).repeat_interleave(args["h"], dim=0)
    # assert torch.allclose(m[0,:], m1[0,0,:])
    # assert torch.allclose(m[0,:], m2[0,0,:])
    # assert not torch.allclose(m[1,:], m1[1,0,:])
    # assert not torch.allclose(m[1,:], m2[1,0,:])

    # Standard attention
    standard_attn, standard_weights = pytorch_scaled_dot_product_attention(
        query=args["q"],
        key=args["k"],
        value=args["v"],
        attn_mask=args["src_key_padding_mask"].unsqueeze(1).repeat_interleave(args["h"], dim=0),
        dropout_p=args["dropout"],
        is_causal=False,
    )

    # Denoising attention (training)
    denoising_attn, denoising_weights = denoising_attention_train(
        query=args["q"],
        key=args["d_k"],
        value=args["d_v"],
        Z=args["latent_dict"]["z"][0],
        pi=args["latent_dict"]["pi"],
        attn_mask=args["latent_dict"]["memory_key_padding_mask"]
        .unsqueeze(1)
        .repeat_interleave(args["h"], dim=0),
        dropout_p=args["dropout"],
    )

    # Check equivalence
    assert torch.allclose(standard_attn, denoising_attn, atol=args["atol"])
    assert torch.allclose(standard_weights, denoising_weights[:, :, 1:], atol=args["atol"])


# Evaluation function
def test_self_attention_eval(seed=42, **kwargs):
    args = random_model_initialisation(seed, is_cross=False, **kwargs)

    # m = args["src_key_padding_mask"]
    # m1 = m.unsqueeze(1).repeat(args["h"], 1, 1) # THIS IS WRONG
    # m2 = args["src_key_padding_mask"].unsqueeze(1).repeat_interleave(args["h"], dim=0)
    # assert torch.allclose(m[0,:], m1[0,0,:])
    # assert torch.allclose(m[0,:], m2[0,0,:])
    # assert not torch.allclose(m[1,:], m1[1,0,:])
    # assert not torch.allclose(m[1,:], m2[1,0,:])

    # Standard attention
    standard_attn, standard_weights = pytorch_scaled_dot_product_attention(
        query=args["q"],
        key=args["k"],
        value=args["v"],
        attn_mask=args["src_key_padding_mask"].unsqueeze(1).repeat_interleave(args["h"], dim=0),
        dropout_p=args["dropout"],
        is_causal=False,
    )

    q_reshape = args["q"].view(
        args["b"], args["h"], args["Nt"], args["head_dim"]
    )  # [B, heads, Nt, d/head]

    # Project the multihead query into the p space from the e (d/head) space
    projected_u = torch.einsum("bhme, hep -> bhmp", q_reshape, args["weights"]["w_k"])
    projected_b = torch.einsum("bhme, hei -> bhmi", q_reshape, args["bias"]["b_k"])  # i = 1

    # Denoising attention (training)
    denoising_attn, denoising_weights = denoising_attention_eval(
        projected_u=projected_u,
        projected_b=projected_b,
        mu=args["latent_dict"]["mu"],
        logvar=args["latent_dict"]["logvar"],
        pi=args["latent_dict"]["pi"],
        w_v=args["weights"]["w_v"],
        b_v=args["bias"]["b_v"],
        attn_mask=args["latent_dict"]["memory_key_padding_mask"]
        .unsqueeze(1)
        .repeat_interleave(args["h"], dim=0),
        dropout_p=args["dropout"],
    )

    # Check equivalence
    assert torch.allclose(standard_attn, denoising_attn, atol=args["atol"])
    assert torch.allclose(standard_weights, denoising_weights[:, :, 1:], atol=args["atol"])


def test_causal_self_attention_train(seed, **kwargs):
    args = random_model_initialisation(seed, is_cross=False, **kwargs)

    # m = args["src_key_padding_mask"]
    # m1 = m.unsqueeze(1).repeat(args["h"], 1, 1) # THIS IS WRONG
    # m2 = args["src_key_padding_mask"].unsqueeze(1).repeat_interleave(args["h"], dim=0)
    # assert torch.allclose(m[0,:], m1[0,0,:])
    # assert torch.allclose(m[0,:], m2[0,0,:])
    # assert not torch.allclose(m[1,:], m1[1,0,:])
    # assert not torch.allclose(m[1,:], m2[1,0,:])

    # Standard attention
    standard_attn, standard_weights = pytorch_scaled_dot_product_attention(
        query=args["q"],
        key=args["k"],
        value=args["v"],
        attn_mask=args["src_key_padding_mask"].unsqueeze(1).repeat_interleave(args["h"], dim=0),
        dropout_p=args["dropout"],
        is_causal=True,
    )

    # Denoising attention (training)
    denoising_attn, denoising_weights = denoising_attention_train(
        query=args["q"],
        key=args["d_k"],
        value=args["d_v"],
        Z=args["latent_dict"]["z"][0],
        pi=args["latent_dict"]["pi"],
        attn_mask=args["latent_dict"]["memory_key_padding_mask"]
        .unsqueeze(1)
        .repeat_interleave(args["h"], dim=0),
        dropout_p=args["dropout"],
        is_causal=True,
    )

    # Check equivalence
    assert torch.allclose(standard_attn, denoising_attn, atol=args["atol"])
    assert torch.allclose(standard_weights, denoising_weights[:, :, 1:], atol=args["atol"])


def test_causal_self_attention_eval(seed, **kwargs):
    args = random_model_initialisation(seed, is_cross=False, **kwargs)

    # Standard attention
    standard_attn, standard_weights = pytorch_scaled_dot_product_attention(
        query=args["q"],
        key=args["k"],
        value=args["v"],
        attn_mask=args["src_key_padding_mask"].unsqueeze(1).repeat_interleave(args["h"], dim=0),
        dropout_p=args["dropout"],
        is_causal=True,
    )

    q_reshape = args["q"].view(
        args["b"], args["h"], args["Nt"], args["head_dim"]
    )  # [B, heads, Nt, d/head]

    # Project the multihead query into the p space from the e (d/head) space
    projected_u = torch.einsum("bhme, hep -> bhmp", q_reshape, args["weights"]["w_k"])
    projected_b = torch.einsum("bhme, hei -> bhmi", q_reshape, args["bias"]["b_k"])  # i = 1

    # Denoising attention (training)
    denoising_attn, denoising_weights = denoising_attention_eval(
        projected_u=projected_u,
        projected_b=projected_b,
        mu=args["latent_dict"]["mu"],
        logvar=args["latent_dict"]["logvar"],
        pi=args["latent_dict"]["pi"],
        w_v=args["weights"]["w_v"],
        b_v=args["bias"]["b_v"],
        attn_mask=args["latent_dict"]["memory_key_padding_mask"]
        .unsqueeze(1)
        .repeat_interleave(args["h"], dim=0),
        dropout_p=args["dropout"],
        is_causal=True,
    )

    # Check equivalence
    assert torch.allclose(standard_attn, denoising_attn, atol=args["atol"])
    assert torch.allclose(standard_weights, denoising_weights[:, :, 1:], atol=args["atol"])


if __name__ == "__main__":
    # test_cross_attention_eval()
    for i in tqdm(range(0, 10)):
        # test_cross_attention_train(i, h=4, Nt=3, Ns=5, b=2)
        test_cross_attention_train(i)
        test_cross_attention_eval(i)
        test_self_attention_train(i)
        test_self_attention_eval(i)
        test_causal_self_attention_train(i)
        test_causal_self_attention_eval(i)
