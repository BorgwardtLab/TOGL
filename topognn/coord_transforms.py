import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Triangle_transform(nn.Module):
    def __init__(self, output_dim):
        """
        output dim is the number of t parameters in the triangle point transformation
        """
        super().__init__()

        self.output_dim = output_dim
        self.t_param = torch.nn.Parameter(
            torch.randn(output_dim)*0.1, requires_grad=True)

    def forward(self, x):
        """
        x is of shape [N,2]
        output is of shape [N,output_dim]
        """

        return torch.nn.functional.relu(x[:, 1][:, None] - torch.abs(self.t_param-x[:, 0][:, None]))


def batch_to_tensor(batch, external_tensor, attribute='x'):
    """
    Takes a pytorch geometric batch and returns the data as a regular tensor padded with 0 and the associated mask
    stacked_tensor [Num graphs, Max num nodes, D]
    mask [Num_graphs, Max num nodes]
    """

    batch_list = []
    idx = batch.__slices__[attribute]

    for i in range(1, 1+len(batch.y)):
        batch_list.append(external_tensor[idx[i-1]:idx[i]])

    stacked_tensor = torch.nn.utils.rnn.pad_sequence(
        batch_list, batch_first=True)  # .permute(1,0,2)
    mask = torch.zeros(stacked_tensor.shape[:2])

    for i in range(1, 1+len(batch.y)):
        mask[i-1, :(idx[i]-idx[i-1])] = 1

    mask_zeros = (stacked_tensor != 0).any(2)
    return stacked_tensor, mask.to(bool), mask_zeros.to(bool)


class Gaussian_transform(nn.Module):

    def __init__(self, output_dim):
        """
        output dim is the number of t parameters in the Gaussian point transformation
        """
        super().__init__()

        self.output_dim = output_dim
        self.t_param = torch.nn.Parameter(
            torch.randn(output_dim)*0.1, requires_grad=True)
        self.sigma = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        """
        x is of shape [N,2]
        output is of shape [N,output_dim]
        """
        return torch.exp(- (x[:, :, None]-self.t_param).pow(2).sum(axis=1) / (2*self.sigma.pow(2)))


class Line_transform(nn.Module):

    def __init__(self, output_dim):
        """
        output dim is the number of lines in the Line point transformation
        """
        super().__init__()

        self.output_dim = output_dim

        self.lin_mod = torch.nn.Linear(2, output_dim)

    def forward(self, x):
        """
        x is of shape [N,2]
        output is of shape [N,output_dim]
        """
        return self.lin_mod(x)


class RationalHat_transform(nn.Module):
    """
    Coordinate function as defined in 

    /Hofer, C., Kwitt, R., and Niethammer, M.
    Learning representations of persistence barcodes.
    JMLR, 20(126):1–45, 2019b./

    """

    def __init__(self, output_dim, input_dim = 1):
        """
        output dim is the number of lines in the Line point transformation
        """
        super().__init__()

        self.output_dim = output_dim

        self.c_param = torch.nn.Parameter(
            torch.randn(input_dim, output_dim)*0.1, requires_grad=True)
        self.r_param = torch.nn.Parameter(
            torch.randn(1, output_dim)*0.1, requires_grad=True)

    def forward(self, x):
        """
        x is of shape [N,input_dim]
        output is of shape [N,output_dim]
        """

        first_element = 1+torch.norm(x[:, :, None]-self.c_param, p=1, dim=1)
        second_element = 1 + \
            torch.abs(torch.abs(self.r_param) -
                      torch.norm(x[:, :, None]-self.c_param, p=1, dim=1))

        return (1/first_element) - (1/second_element)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)  # * num_heads)
        self.fc_k = nn.Linear(dim_K, dim_V)  # * num_heads)
        self.fc_v = nn.Linear(dim_K, dim_V)  # * num_heads)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask=None):
        """
        mask should be of shape [batch, length]
        """

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # Modification to handle masking.
        if mask is not None:
            mask_repeat = mask[:, None, :].repeat(
                self.num_heads, Q.shape[1], 1)
            before_softmax = Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V)
            before_softmax[~mask_repeat] = -1e10
        else:
            before_softmax = Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V)

        A = torch.softmax(before_softmax, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, mask):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask)
        return self.mab1(X, H)


class Set2SetMod(torch.nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds):
        super().__init__()
        self.set_transform = ISAB(dim_in=dim_in,
                                  dim_out=dim_out,
                                  num_heads=num_heads,
                                  num_inds=num_inds)

    def forward(self, x, batch, dim1_flag=False):

        if dim1_flag:
            stacked_tensor, mask, mask_zeros = batch_to_tensor(
                batch, x, attribute="edge_index")
            out_ = self.set_transform(stacked_tensor, mask)
            out_[mask_zeros] = 0
            out = out_[mask]
        else:
            stacked_tensor, mask, mask_zeros = batch_to_tensor(batch, x)
            out_ = self.set_transform(stacked_tensor, mask)
            out = out_[mask]

        return out


#mod = ISAB(dim_in = 2, dim_out = 32, num_heads = 4, num_inds = 6, ln = False)

#x = torch.randn((2,12,2))

#y = mod(x, mask = torch.randint(high=2, size = (2,12)))
