import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Triangle_transform(nn.Module):
    def __init__(self,output_dim):
        """
        output dim is the number of t parameters in the triangle point transformation
        """
        super().__init__()

        self.output_dim = output_dim
        self.t_param = torch.nn.Parameter(torch.zeros(output_dim),requires_grad = True)

    def forward(self,x):
        """
        x is of shape [N,2]
        output is of shape [N,output_dim]
        """

        return torch.nn.functional.relu( x[:,1][:,None] - torch.abs(self.t_param-x[:,0][:,None] ) )


class Gaussian_transform(nn.Module):

    def __init__(self,output_dim):
        """
        output dim is the number of t parameters in the Gaussian point transformation
        """
        super().__init__()

        self.output_dim = output_dim
        self.t_param = torch.nn.Parameter(torch.zeros(output_dim),requires_grad = True)
        self.sigma = torch.nn.Parameter(torch.ones(1),requires_grad = True)

    def forward(self,x):
        """
        x is of shape [N,2]
        output is of shape [N,output_dim]
        """
        return torch.exp(- (x[:,:,None]-self.t_param).pow(2).sum(axis=1) / (2*self.sigma.pow(2)))

class Line_transform(nn.Module):

    def __init__(self,output_dim):
        """
        output dim is the number of lines in the Line point transformation
        """
        super().__init__()

        self.output_dim = output_dim

        self.lin_mod = torch.nn.Linear(2,output_dim)

    def forward(self,x):
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

    def __init__(self,output_dim):

        """
        output dim is the number of lines in the Line point transformation
        """
        super().__init__()

        self.output_dim = output_dim

        self.c_param = torch.nn.Parameter(torch.zeros(2,output_dim),requires_grad = True)
        self.r_param = torch.nn.Parameter(torch.zeros(1,output_dim),requires_grad = True)

    def forward(self,x):
        """
        x is of shape [N,2]
        output is of shape [N,output_dim]
        """

        first_element = 1+torch.norm(x[:,:,None]-self.c_param,p=1,dim=1)
        second_element = 1+torch.abs(torch.abs(self.r_param)-torch.norm(x[:,:,None]-self.c_param,p=1,dim=1))

        return (1/first_element) - (1/second_element)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V )#* num_heads)
        self.fc_k = nn.Linear(dim_K, dim_V )#* num_heads)
        self.fc_v = nn.Linear(dim_K, dim_V )#* num_heads)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V , dim_V)

    def forward(self, Q, K, mask = None):
        """
        mask should be of shape [batch, length]
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        #Modification to handle masking.
        if mask is not None:
            mask_repeat = mask[:,None,:].repeat(self.num_heads,Q.shape[1],1)
            before_softmax = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
            before_softmax[mask_repeat==0] = -1e10 
        else:
            before_softmax = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)

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



#mod = ISAB(dim_in = 2, dim_out = 32, num_heads = 4, num_inds = 6, ln = False)

#x = torch.randn((2,12,2))

#y = mod(x, mask = torch.randint(high=2, size = (2,12)))


