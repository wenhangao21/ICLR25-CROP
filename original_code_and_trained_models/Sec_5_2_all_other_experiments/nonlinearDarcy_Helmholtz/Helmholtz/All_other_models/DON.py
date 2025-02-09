import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import format_tensor_size

def torch2dgrid(num_x, num_y, bot=(0,0), top=(1,1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = torch.linspace(x_bot, x_top, steps=num_x)
    y_arr = torch.linspace(y_bot, y_top, steps=num_y)
    xx, yy = torch.meshgrid(x_arr, y_arr, indexing='ij')
    mesh = torch.stack([xx, yy], dim=2)
    return mesh

class ResNetBlock(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 kernel_size=3):
        super(ResNetBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.cont_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                   kernel_size=self.kernel_size, stride=1,
                                   padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
        self.sigma = F.leaky_relu

    def forward(self, x):
        p = 0.1
        return self.cont_conv(self.sigma(x)) * p + (1 - p) * x

class ConvBranch2D(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 N_layers,  # Number of layers in the network
                 N_res,
                 out_channel=1,
                 kernel_size=3,
                 multiply=32,
                 print_bool=False
                 ):  # Number of ResNet Blocks

        super(ConvBranch2D, self).__init__()

        assert N_layers % 2 == 0, "Number of layers myst be even number."

        self.N_layers = N_layers
        self.print_bool = print_bool
        self.channel_multiplier = multiply
        self.feature_maps = [in_channels]
        for i in range(0, self.N_layers):
            self.feature_maps.append(2 ** i * self.channel_multiplier)
        self.feature_maps_invariant = self.feature_maps

        print("channels: ", self.feature_maps)

        assert len(self.feature_maps) == self.N_layers + 1

        self.kernel_size = kernel_size
        self.cont_conv_layers = nn.ModuleList([nn.Conv2d(self.feature_maps[i],
                                                         self.feature_maps[i + 1],
                                                         kernel_size=self.kernel_size, stride=1,
                                                         padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
                                               for i in range(N_layers)])

        self.cont_conv_layers_invariant = nn.ModuleList([nn.Conv2d(self.feature_maps_invariant[i],
                                                                   self.feature_maps_invariant[i],
                                                                   kernel_size=self.kernel_size, stride=1,
                                                                   padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
                                                         for i in range(N_layers)])

        self.sigma = F.leaky_relu

        self.resnet_blocks = []

        for i in range(N_res):
            self.resnet_blocks.append(ResNetBlock(in_channels=self.feature_maps[self.N_layers]))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        self.N_res = N_res

        self.upsample4 = nn.Upsample(scale_factor=4)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.downsample2 = nn.AvgPool2d(2, stride=2, padding=0)
        self.downsample4 = nn.AvgPool2d(4, stride=4, padding=1)

        self.flatten_layer = nn.Flatten()
        self.lazy_linear = nn.LazyLinear(out_channel)

    def forward(self, x):
        for i in range(self.N_layers):
            if self.print_bool: print("BEFORE I1", x.shape)
            y = self.cont_conv_layers_invariant[i](x)
            if self.print_bool: print("After I1", y.shape)
            y = self.sigma(self.upsample2(y))
            if self.print_bool: print("After US1", y.shape)
            x = self.downsample2(y)
            if self.print_bool: print("AFTER IS1", x.shape)

            if self.print_bool: print("INV DONE")
            y = self.cont_conv_layers[i](x)
            if self.print_bool: print("AFTER CONTCONV", y.shape)
            y = self.upsample2(y)
            if self.print_bool: print("AFTER UP", y.shape)
            x = self.downsample4(self.sigma(y))
            if self.print_bool: print("AFTER IS2", x.shape)

        for i in range(self.N_res):
            x = self.resnet_blocks[i](x)
            if self.print_bool: print("RES", x.shape)

        x = self.flatten_layer(x)
        if self.print_bool: print("Flattened", x.shape)
        x = self.lazy_linear(x)
        if self.print_bool: print("Linearized", x.shape)
        if self.print_bool: quit()
        return x


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
        self.branch = ConvBranch2D(in_channels=1,  # Number of input channels.
                              N_layers=2,
                              N_res=2,
                              kernel_size=3,
                              multiply=32,
                              out_channel=250)
        self.trunk = DenseNet(trunk_layer, nn.ReLU)

    def forward(self, u0, grid):
        a = self.branch(u0)
        #print(a.shape)
        # batchsize x width
        b = self.trunk(grid)
        #print(b.shape)
        # N x width
        return torch.einsum('bi,ni->bn', a, b)
        