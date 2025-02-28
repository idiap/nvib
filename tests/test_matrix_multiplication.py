import torch

b, n, m, d = 4, 15, 7, 16

Z = torch.rand(b, n, d)
U = torch.rand(b, m, d)

W_q = torch.rand(d, d)
W_k = torch.rand(d, d)
W_v = torch.rand(d, d)

b_q = torch.rand(1, d)
b_k = torch.rand(1, d)
b_v = torch.rand(1, d)

Q = U @ W_q + b_q
K = Z @ W_k + b_k
V = Z @ W_q + b_v

A = torch.einsum("bmd, bnd -> bmn", Q, K)

A_w = torch.softmax(A, dim=-1)

out = torch.einsum("bmn, bnd -> bmd", A_w, V)

# Can we break Z and then project?


breakpoint()
# Multihead
h = 2
head_dim = int(d / h)

Q_mh = Q.transpose(0, 1).view(m, b, h, head_dim).permute(1, 2, 0, 3)
K_mh = K.transpose(0, 1).view(n, b, h, head_dim).permute(1, 2, 0, 3)
V_mh = V.transpose(0, 1).view(n, b, h, head_dim).permute(1, 2, 0, 3)


A_mh = torch.einsum("bhmp, bhnp -> bhmn", Q_mh, K_mh)

A_wmh = torch.softmax(A_mh, dim=-1)

out_mh = torch.einsum("bhmn, bhnd -> bhmd", A_wmh, V_mh)

# Now we want to prove that we can break the weight matrix and its equivalent


def test_multihead_weight_split():
    W_qmh = W_q.view(d, h, head_dim)
    b_qmh = b_q.view(h, head_dim)

    new_Q_mh_nobias = torch.einsum("bmd, dhp -> bhmp", Q, W_qmh)
    new_Q_mh_withbias = torch.einsum("bhmp, hp -> bhmp", new_Q_mh_nobias, b_qmh)

    assert torch.allclose(Q_mh, new_Q_mh_withbias)


if __name__ == "__main__":
    test_multihead_weight_split()
