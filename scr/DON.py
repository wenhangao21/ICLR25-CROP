import torch
import torch.nn as nn
from .utils import format_tensor_size


def torch2dgrid(num_x, num_y, bot=(0,0), top=(1,1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = torch.linspace(x_bot, x_top, steps=num_x)
    y_arr = torch.linspace(y_bot, y_top, steps=num_y)
    xx, yy = torch.meshgrid(x_arr, y_arr, indexing='ij')
    mesh = torch.stack([xx, yy], dim=2)
    return mesh


class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        if isinstance(nonlinearity, str):
            if nonlinearity == 'tanh':
                nonlinearity = nn.Tanh
            elif nonlinearity == 'relu':
                nonlinearity == nn.ReLU
            else:
                raise ValueError(f'{nonlinearity} is not supported')
        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class DeepONet(nn.Module):
    def __init__(self, branch_layer, trunk_layer):
        super(DeepONet, self).__init__()
        self.branch = DenseNet(branch_layer, nn.ReLU)
        self.trunk = DenseNet(trunk_layer, nn.ReLU)

    def forward(self, u0, grid):
        a = self.branch(u0)
        b = self.trunk(grid)
        batchsize = a.shape[0]
        dim = a.shape[1]
        return torch.bmm(a.view(batchsize, 1, dim), b.view(batchsize, dim, 1))


class DeepONetCP(nn.Module):
    def __init__(self, branch_layer, trunk_layer):
        super(DeepONetCP, self).__init__()
        self.branch = DenseNet(branch_layer, nn.ReLU)
        self.trunk = DenseNet(trunk_layer, nn.ReLU)

    def forward(self, u0, grid):
        a = self.branch(u0)
        # batchsize x width
        b = self.trunk(grid)
        # N x width
        return torch.einsum('bi,ni->bn', a, b)
        
    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters in DON: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams