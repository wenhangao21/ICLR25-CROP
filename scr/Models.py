# The CNO2d code has been modified from a tutorial featured in the 
# ETH Zurich course "AI in the Sciences and Engineering."
# Git page for this course: https://github.com/bogdanraonic3/AI_Science_Engineering 

# For up/downsampling, the antialias interpolation functions from the 
# torch library are utilized, limiting the ability to design
# your own low-pass filters at present.

# While acknowledging this suboptimal setup, the performance of CNO2d remains commendable. 
# Additionally, a training script is available, offering a solid foundation for personal projects.


import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
import math
from collections import OrderedDict
from .utils import format_tensor_size


################################################# CNO 2D ######################################     

# CNO LReLu activation fucntion
# CNO building block (CNOBlock) → Conv2d - BatchNorm - Activation
# Lift/Project Block (Important for embeddings)
# Residual Block → Conv2d - BatchNorm - Activation - Conv2d - BatchNorm - Skip Connection
# ResNet → Stacked ResidualBlocks (several blocks applied iteratively)


#---------------------
# Activation Function:
#---------------------

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
        x = F.interpolate(x, size = (2 * self.in_size, 2 * self.in_size), mode = "bicubic", antialias = True)
        x = self.act(x)
        x = F.interpolate(x, size = (self.out_size,self.out_size), mode = "bicubic", antialias = True)
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

#--------------------
# CNO:
#--------------------

class CNO2d(nn.Module):
    def __init__(self,
                in_dim,                    # Number of input channels.
                out_dim,                   # Number of input channels.
                size,                      # Input and Output spatial size (required )
                N_layers,                  # Number of (D) or (U) blocks in the network
                N_res = 4,                 # Number of (R) blocks per level (except the neck)
                N_res_neck = 4,            # Number of (R) blocks in the neck
                channel_multiplier = 16,   # How the number of channels evolve?
                use_bn = True,             # Add BN? We do not add BN in lifting/projection layer
                ):

        super(CNO2d, self).__init__()

        self.N_layers = int(N_layers)         # Number od (D) & (U) Blocks
        self.lift_dim = channel_multiplier//2 # Input is lifted to the half of channel_multiplier dimension
        self.in_dim   = in_dim
        self.out_dim  = out_dim
        self.channel_multiplier = channel_multiplier  # The growth of the channels

        ######## Num of channels/features - evolution ########

        self.encoder_features = [self.lift_dim] # How the features in Encoder evolve (number of features)
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i *   self.channel_multiplier)

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
            self.encoder_sizes.append(size // 2 ** i)
            self.decoder_sizes.append(size // 2 ** (self.N_layers - i))
        #print(self.encoder_sizes)
        #print(self.decoder_sizes)

        ######## Define Lift and Project blocks ########

        self.lift   = LiftProjectBlock(in_channels = in_dim,
                                        out_channels = self.encoder_features[0],
                                        size = size)

        self.project   = LiftProjectBlock(in_channels = self.encoder_features[0] + self.decoder_features_out[-1],
                                            out_channels = out_dim,
                                            size = size)

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
    
    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters in CNO: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams
        
        
################################################# FNO 2D Time ######################################        
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
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
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
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        self.get_weight()
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[..., (freq0_y - self.modes1 + 1):(freq0_y + self.modes1), :self.modes2]
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)              
        out_ft[..., (freq0_y - self.modes1 + 1):(freq0_y + self.modes1), :self.modes2] = \
            self.compl_mul2d(x_ft, self.weights)


        # Return to physical space
        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1)))
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

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_layers=4, in_channel_dim=1, out_channel_dim=1, grid = True):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains `num_layers` layers of the Fourier layer.
        1. Lift the input to the desired channel dimension by self.fc0.
        2. `num_layers` layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv.
        3. Project from the channel space to the output space by self.fc1 and self.fc2.

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8  # pad the domain if input is non-periodic
        self.num_layers = num_layers
        self.in_channel_dim = in_channel_dim
        self.out_channel_dim = out_channel_dim
        self.grid = grid
        if grid:
            # self.p = nn.Linear(self.in_channel_dim + 2, self.width)  # input channel is 1
            self.p = nn.Conv2d(self.in_channel_dim + 2, self.width, 1)
        else:
            #self.p = nn.Linear(self.in_channel_dim, self.width)
            self.p = nn.Conv2d(self.in_channel_dim , self.width, 1)

        # Create lists of conv, mlp, and w layers based on num_layers
        self.conv_layers = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(num_layers)])
        self.mlp_layers = nn.ModuleList([MLP(self.width, self.width, self.width) for _ in range(num_layers)])
        self.w_layers = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(num_layers)])

        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, self.out_channel_dim, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):
        if self.grid:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)
        x = self.p(x)

        for i in range(self.num_layers):
            x1 = self.norm(self.conv_layers[i](self.norm(x)))
            x1 = self.mlp_layers[i](x1)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)

        x = self.q(x)
        #x = x.permute(0, 2, 3, 1)
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

        print(f'Total number of model parameters in FNO: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams
        
        
################### UNet From PDEBench ####################
class UNet2d(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet2d, self).__init__()

        features = init_features
        self.encoder1 = UNet2d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet2d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet2d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet2d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        #print(enc1.shape) #85
        enc2 = self.encoder2(self.pool1(enc1))
        #print(enc2.shape) #42
        enc3 = self.encoder3(self.pool2(enc2))
        #print(enc3.shape) # 21
        enc4 = self.encoder4(self.pool3(enc3))
        #print(enc4.shape) # 10

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        #print(dec4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        #print(dec4.shape)
        dec4 = self.decoder4(dec4)
        #print(dec4.shape)
        dec3 = self.upconv3(dec4)
        #print(dec3.shape)
        dec3 = torch.cat((dec3, enc3), dim=1)
        #print(dec3.shape)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out = self.conv(dec1)
        return out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )
        
    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters in FNO: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams