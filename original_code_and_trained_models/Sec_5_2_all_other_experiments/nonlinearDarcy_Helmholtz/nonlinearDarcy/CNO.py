import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils import format_tensor_size


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
        x = x.permute(0, 3, 1, 2)
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
        x = x.permute(0, 2, 3, 1)

        return x
    
    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters in CNO: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams