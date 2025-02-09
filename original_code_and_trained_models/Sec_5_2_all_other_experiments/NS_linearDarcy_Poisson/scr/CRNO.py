import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
import math

from .utils import format_tensor_size


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

        # get the index of the zero frequency and construct weight
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        self.get_weight()
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x, norm = "ortho"), dim=-2)
        #print("x_ft:",x_ft.shape)
        x_ft = x_ft[..., (freq0_y - self.modes1 + 1):(freq0_y + self.modes1), :self.modes2]
        # Multiply relevant Fourier modes
        #print("x_ft:",x_ft.shape)
        out_ft = torch.zeros(batchsize, self.out_channels, self.out_size, self.out_size // 2 + 1, dtype=torch.cfloat, device=x.device)
        #print("out_ft:",out_ft.shape)
        #print("self.weights:",self.weights.shape)   
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(self.out_size)) == 0).nonzero().item()
        out_ft[..., (freq0_y - self.modes1 + 1):(freq0_y + self.modes1), :self.modes2] = self.compl_mul2d(x_ft, self.weights) 


        # Return to physical space
        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), norm = "ortho") * (self.out_size/in_size)
        return x

class CNO_LReLu(nn.Module):
    def __init__(self,
                in_size,
                out_size
                ):
        super(CNO_LReLu, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.act = nn.LeakyReLU()

    def forward(self, x):
        #x = F.interpolate(x, size = (2 * self.in_size, 2 * self.in_size), mode = "bicubic", antialias = True)
        x = self.act(x)
        if self.in_size != self.out_size:
            x = F.interpolate(x, size = (self.out_size,self.out_size), mode = "bicubic", antialias = False)
        return x

#--------------------
# CNO Block:
#--------------------

class CNOBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                in_size,
                out_size,
                use_bn = True
                ):
        super(CNOBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size  = in_size
        self.out_size = out_size

        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation

        self.convolution = torch.nn.Conv2d(in_channels = self.in_channels,
                                            out_channels= self.out_channels,
                                            kernel_size = 3,
                                            padding     = 1)

        if use_bn:
            self.batch_norm  = nn.BatchNorm2d(self.out_channels)
        else:
            self.batch_norm  = nn.Identity()
        self.act           = CNO_LReLu(in_size  = self.in_size,
                                        out_size = self.out_size)
    def forward(self, x):
        x = self.convolution(x)
        x = self.batch_norm(x)
        return self.act(x)
    
#--------------------
# Lift/Project Block:
#--------------------

class LiftProjectBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                size,
                latent_dim = 64
                ):
        super(LiftProjectBlock, self).__init__()

        self.inter_CNOBlock = CNOBlock(in_channels       = in_channels,
                                        out_channels     = latent_dim,
                                        in_size          = size,
                                        out_size         = size,
                                        use_bn           = False)

        self.convolution = torch.nn.Conv2d(in_channels  = latent_dim,
                                            out_channels = out_channels,
                                            kernel_size  = 3,
                                            padding      = 1)


    def forward(self, x):
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
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
        #print(x.shape)
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

#--------------------
# Residual Block:
#--------------------

class ResidualBlock(nn.Module):
    def __init__(self,
                channels,
                size,
                use_bn = True
                ):
        super(ResidualBlock, self).__init__()

        self.channels = channels
        self.size     = size

        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation -> Conv -> BN (optional) -> Skip Connection
        # Up/Downsampling happens inside Activation

        self.convolution1 = torch.nn.Conv2d(in_channels = self.channels,
                                            out_channels= self.channels,
                                            kernel_size = 3,
                                            padding     = 1)
        self.convolution2 = torch.nn.Conv2d(in_channels = self.channels,
                                            out_channels= self.channels,
                                            kernel_size = 3,
                                            padding     = 1)

        if use_bn:
            self.batch_norm1  = nn.BatchNorm2d(self.channels)
            self.batch_norm2  = nn.BatchNorm2d(self.channels)

        else:
            self.batch_norm1  = nn.Identity()
            self.batch_norm2  = nn.Identity()

        self.act           = CNO_LReLu(in_size  = self.size,
                                        out_size = self.size)


    def forward(self, x):
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)
        return x + out

#--------------------
# ResNet:
#--------------------

class ResNet(nn.Module):
    def __init__(self,
                channels,
                size,
                num_blocks,
                use_bn = True
                ):
        super(ResNet, self).__init__()

        self.channels = channels
        self.size = size
        self.num_blocks = num_blocks

        self.res_nets = []
        for _ in range(self.num_blocks):
            self.res_nets.append(ResidualBlock(channels = channels,
                                                size = size,
                                                use_bn = use_bn))

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        #print(x.shape)
        for i in range(self.num_blocks):
            x = self.res_nets[i](x)
            #print(x.shape)
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

#--------------------
# CNO:
#--------------------

class CRNO2d(nn.Module):
    def __init__(self,
                in_dim,                    # Number of input channels.
                out_dim,                   # Number of input channels.
                in_out_size,               # Input and output spacial size
                latent_size,               # Latent spacial size
                modes,
                N_layers,                  # Number of (D) or (U) blocks in the network
                N_res = 4,                 # Number of (R) blocks per level (except the neck)
                N_res_neck = 6,            # Number of (R) blocks in the neck
                ini_channel = 32,   # How the number of channels evolve?
                use_bn = True,             # Add BN? We do not add BN in lifting/projection layer
                ):

        super(CRNO2d, self).__init__()

        self.N_layers = int(N_layers)         # Number od (D) & (U) Blocks
        self.lift_dim = ini_channel//2 # Input is lifted to the half of channel_multiplier dimension
        self.in_dim   = in_dim
        self.out_dim  = out_dim
        self.ini_channel  = ini_channel  # The growth of the channels
        self.in_out_size = in_out_size
        self.latent_size = latent_size
        self.modes = modes

        ######## Num of channels/features - evolution ########

        self.encoder_features = [self.lift_dim] # How the features in Encoder evolve (number of features)
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i *   self.ini_channel)

        self.decoder_features_in = self.encoder_features[1:] # How the features in Decoder evolve (number of features)
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2*self.decoder_features_in[i] #Pad the outputs of the resnets (we must multiply by 2 then)
        
        #print(self.encoder_features)
        #print(self.decoder_features_in)
        #print(self.decoder_features_out)
        ######## Spatial sizes of channels - evolution ########

        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(latent_size // 2 ** i)
            self.decoder_sizes.append(latent_size // 2 ** (self.N_layers - i))
        #print(self.encoder_sizes)
        #print(self.decoder_sizes)

        ######## Define Lift and Project blocks ########

        self.lift   = LiftBlock(in_channels = in_dim + 2,
                                        out_channels = self.encoder_features[0],
                                        out_size = latent_size,
                                        modes1 = modes,
                                        modes2 = modes)

        self.project   = ProjectBlock(in_channels = self.encoder_features[0] + self.decoder_features_out[-1],
                                            out_channels = out_dim,
                                            out_size = in_out_size,
                                            modes1 = modes,
                                            modes2 = modes)

        ######## Define Encoder, ED Linker and Decoder networks ########

        self.encoder         = nn.ModuleList([(CNOBlock(in_channels  = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i+1],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.encoder_sizes[i+1],
                                                        use_bn       = use_bn))
                                                for i in range(self.N_layers)])

        # After the ResNets are executed, the sizes of encoder and decoder might not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        self.ED_expansion     = nn.ModuleList([(CNOBlock(in_channels = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.decoder_sizes[self.N_layers - i],
                                                        use_bn       = use_bn))
                                                for i in range(self.N_layers + 1)])

        self.decoder         = nn.ModuleList([(CNOBlock(in_channels  = self.decoder_features_in[i],
                                                        out_channels = self.decoder_features_out[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i+1],
                                                        use_bn       = use_bn))
                                                for i in range(self.N_layers)])

        #### Define ResNets Blocks 

        # Here, we define ResNet Blocks.

        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding sets of feature maps in the decoder

        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        # Define the ResNet networks (before the neck)
        for l in range(self.N_layers):
            self.res_nets.append(ResNet(channels = self.encoder_features[l],
                                        size = self.encoder_sizes[l],
                                        num_blocks = self.N_res,
                                        use_bn = use_bn))

        self.res_net_neck = ResNet(channels = self.encoder_features[self.N_layers],
                                    size = self.encoder_sizes[self.N_layers],
                                    num_blocks = self.N_res_neck,
                                    use_bn = use_bn)

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
    
        #print("1: ", x.shape)        
        x = self.lift(x) #Execute Lift
        #print("2: ", x.shape)   
        skip = []
       
        # Execute Encoder
        for i in range(self.N_layers):

            #Apply ResNet & save the result
            y = self.res_nets[i](x)
            #print("3: ", i, y.shape)   
            skip.append(y)

            # Apply (D) block
            x = self.encoder[i](x)
            #print("4: ", i, x.shape)   
        
        # Apply the deepest ResNet (bottle neck)
        x = self.res_net_neck(x)
        #print("5: ", x.shape)   
        # Execute Decode
        for i in range(self.N_layers):

            # Apply (I) block (ED_expansion) & cat if needed
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x) #BottleNeck : no cat
                #print("6: ", i, x.shape)   
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i])),1)
                #print("6: ", i, x.shape)   
            # Apply (U) block
            x = self.decoder[i](x)
            #print("7: ", i, x.shape)   

        # Cat & Execute Projetion
        x = torch.cat((x, self.ED_expansion[0](skip[0])),1)
        #print("8: ", x.shape)  
        x = self.project(x)
        #print("9: ", x.shape)  
        
        
        
        return x
        
        
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1)
    
    
    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters in CRNO: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams
        
       