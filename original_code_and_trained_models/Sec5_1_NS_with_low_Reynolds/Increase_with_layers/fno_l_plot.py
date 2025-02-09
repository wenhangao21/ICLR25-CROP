import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer
import math


################################################################
# fourier layer
################################################################


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
    def __init__(self, modes1, modes2, width, num_layers=4):
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

        self.p = nn.Linear(3, self.width)  # input channel is 12

        # Create lists of conv, mlp, and w layers based on num_layers
        self.conv_layers = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(num_layers)])
        self.mlp_layers = nn.ModuleList([MLP(self.width, self.width, self.width) for _ in range(num_layers)])
        self.w_layers = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(num_layers)])

        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 1, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.num_layers):
            x1 = self.norm(self.conv_layers[i](self.norm(x)))
            x1 = self.mlp_layers[i](x1)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)

        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)


def trig_inter_32_to_256(u1):
  bsize = u1.shape[0]
  fu1 = torch.fft.rfft2(u1, norm = "ortho")
  fu1_recover = torch.zeros((bsize, 256, 129), dtype=torch.complex64, device=device)
  fu1_recover[:,:32, :33] = fu1[:,:32,:33]
  fu1_recover[:,-32:, :33] = fu1[:,-32:,:33]
  u1_recover = torch.fft.irfft2(fu1_recover, norm = "ortho") * 4
  return u1_recover
  
myloss = LpLoss(size_average=False)
################################################################
# configs
################################################################

TRAIN_PATH = 'your_data_path'
TRAIN_PATH2 = 'your_data_path'

ntrain = 20
ntest = 200

modes = 12
width = 20

batch_size = 10
learning_rate = 0.001
epochs = 500
iterations = epochs*(ntrain//batch_size)

path = 'ns_fourier_2d_time_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

sub = 1
S = 256
T_in = 1
T = 1 # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;
step = 1

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain,::sub,::sub,10:11].to(device)
train_u = reader.read_field('u')[:ntrain,::sub,::sub,11:12].to(device)


reader = MatReader(TRAIN_PATH2)
train_a2 = reader.read_field('u')[:ntrain,::sub,::sub,10:11].to(device)
train_u2 = reader.read_field('u')[:ntrain,::sub,::sub,11:12].to(device)

# print(train_u.shape)
# print(test_u.shape)
# assert (S == train_u.shape[-2])
# assert (T == train_u.shape[-1])

train_a = train_a.reshape(20,S,S,T_in)
train_a2 = train_a2.reshape(20,64,64,T_in)
# test_a = test_a.reshape(ntest,S,S,T_in)

myloss = LpLoss(size_average=False)
################################################################
# training and evaluation
################################################################
for layers in range(1, 16):
    rel_l2_values = []  # List to store L2 values for each seed
    for seed in range(0, 20):
        model = FNO2d(modes, modes, width, layers).cuda()
        model.load_state_dict(torch.load(f"FNO_l{layers}_{seed}.pth"))

        with torch.no_grad():
            im = model(train_a)
            im2 = model(train_a2)
            im = im.squeeze(-1)
            im2 = im2.squeeze(-1)
            im3 = trig_inter_32_to_256(im2)
            rel_l2 = torch.linalg.norm(im3 - im)/torch.linalg.norm(train_u)
            #rel_l2 = torch.linalg.norm(im - train_u.squeeze(-1))/torch.linalg.norm(train_u.squeeze(-1))
            #print(torch.linalg.norm(im2)/64/64)
            rel_l2_values.append(rel_l2)

    rel_l2_tensor = torch.tensor(rel_l2_values)  # Convert list to tensor
    mean_rel_l2 = torch.mean(rel_l2_tensor)  # Calculate mean
    std_rel_l2 = torch.std(rel_l2_tensor)    # Calculate standard deviation

    print(f"Layer {layers}: Mean L2 = {mean_rel_l2}, Std L2 = {std_rel_l2}")


        

