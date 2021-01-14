import torch
import torch.nn as nn

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


#mod = RationalHat_transform(4)

#x = torch.randn((12,2))

#y = mod(x)


