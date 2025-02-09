import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
import math

from utils import format_tensor_size


class Spectral_weights(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        dtype = torch.cfloat
        self.kernel_size_Y = 2*modes1 - 1
        self.kernel_size_X = modes2
        self.W = nn.ParameterDict({
            'y0_modes': torch.nn.Parameter(torch.empty(in_channels, out_channels, modes1 - 1, 1, dtype=dtype)),
            'yposx_modes': torch.nn.Parameter(torch.empty(in_channels, out_channels, self.kernel_size_Y, self.kernel_size_X - 1, dtype=dtype)),
            '00_modes': torch.nn.Parameter(torch.empty(in_channels, out_channels, 1, 1, dtype=torch.float))
        })
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        for v in self.W.values():
            nn.init.kaiming_uniform_(v, a=math.sqrt(5))
            
    def get_weight(self):
        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        self.weights = torch.cat([self.W["y0_modes"], self.W["00_modes"].cfloat(), self.W["y0_modes"].flip(dims=(-2, )).conj()], dim=-2)
        self.weights = torch.cat([self.weights, self.W["yposx_modes"]], dim=-1)
        self.weights = self.weights.view(self.in_channels, self.out_channels,
                                         self.kernel_size_Y, self.kernel_size_X)
                                                                                  
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, out_size):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.out_size = out_size
        self.spectral_weight = Spectral_weights(in_channels=in_channels, out_channels=out_channels, modes1=modes1, modes2=modes2)
        self.get_weight()

    def get_weight(self):
        self.spectral_weight.get_weight()
        self.weights = self.spectral_weight.weights
        
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        #print("x.shape:",x.shape)
        batchsize = x.shape[0]
        in_size = x.shape[-1]
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        self.get_weight()
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x, norm = "ortho"), dim=-2)
        x_ft = x_ft[..., (freq0_y - self.modes1 + 1):(freq0_y + self.modes1), :self.modes2]
        out_ft = torch.zeros(batchsize, self.out_channels, self.out_size, self.out_size // 2 + 1, dtype=torch.cfloat, device=x.device)
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(self.out_size)) == 0).nonzero().item()
        out_ft[..., (freq0_y - self.modes1 + 1):(freq0_y + self.modes1), :self.modes2] = self.compl_mul2d(x_ft, self.weights) 

        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s = (self.out_size, self.out_size), norm = "ortho") * (self.out_size/in_size)
        return x
    

class MLP3(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP3, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels,  3, padding=1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
        
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x    
        

class LiftBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                out_size,
                modes1,
                modes2,
                latent_dim = 16,
                use_bn = True
                ):
        super(LiftBlock, self).__init__()

        self.conv_layer = SpectralConv2d(in_channels, latent_dim, modes1, modes2, out_size)
        self.convolution = torch.nn.Conv2d(in_channels  = latent_dim,
                                                    out_channels = out_channels,
                                                    kernel_size  = 3,
                                                    padding      = 1, 
                                                    #padding_mode = 'circular'
                                                    )
        if use_bn:
            self.batch_norm  = nn.BatchNorm2d(latent_dim)
        else:
            self.batch_norm  = nn.Identity()
        self.act = nn.GELU()
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.convolution(x)
        return x
        
        
class ProjectBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                out_size,
                modes1,
                modes2,
                latent_dim = 20,
                use_bn = True
                ):
        super(ProjectBlock, self).__init__()

        self.conv_layer = SpectralConv2d(in_channels, latent_dim, modes1, modes2, out_size)
        self.q = MLP(latent_dim, out_channels, latent_dim * 4)  
        if use_bn:
            self.batch_norm  = nn.BatchNorm2d(latent_dim)
        else:
            self.batch_norm  = nn.Identity()
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = self.q(x)
        return x


class CRNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_out_size, latent_size):
        super(CRNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.lift   = LiftBlock(in_channels = 3,
                                out_channels = width,
                                out_size = latent_size,
                                modes1 = modes1,
                                modes2 = modes2)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, latent_size)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, latent_size)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, latent_size)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, latent_size)
        self.mlp0 = MLP3(self.width, self.width, self.width)
        self.mlp1 = MLP3(self.width, self.width, self.width)
        self.mlp2 = MLP3(self.width, self.width, self.width)
        self.mlp3 = MLP3(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 3, padding=1)
        self.w1 = nn.Conv2d(self.width, self.width, 3, padding=1)
        self.w2 = nn.Conv2d(self.width, self.width, 3, padding=1)
        self.w3 = nn.Conv2d(self.width, self.width, 3, padding=1)
        self.project   = ProjectBlock(in_channels = width,
                                            out_channels = 1,
                                            out_size = in_out_size,
                                            modes1 = modes1,
                                            modes2 = modes2)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)   
        x = self.lift(x) #Execute Lift
        
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)  # Residual Block
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)  # Residual Block
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)  # Residual Block
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)  # Residual Block
        x = x1 + x2
        
        x = self.project(x)
        x = x.permute(0, 2, 3, 1)
        return x
        
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters in CNO: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams
        
       