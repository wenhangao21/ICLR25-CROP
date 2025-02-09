from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from utilities3 import *

import scipy.io
from utils import UnitGaussianNormalizer, count_params, save_checkpoint, LpLoss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def torch2dgrid(num_x, num_y, bot=(0,0), top=(1,1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = torch.linspace(x_bot, x_top, steps=num_x)
    y_arr = torch.linspace(y_bot, y_top, steps=num_y)
    xx, yy = torch.meshgrid(x_arr, y_arr, indexing='ij')
    mesh = torch.stack([xx, yy], dim=2)
    return mesh

class DenseNet(nn.Module):
    """
    A fully connected neural network (MLP) with ReLU activations between layers, except the last one.
    """
    def __init__(self, layers):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = nn.ModuleList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x


class DeepONet(nn.Module):
    def __init__(self, branch_layer, trunk_layer):
        super(DeepONet, self).__init__()
        self.branch = DenseNet(branch_layer)
        self.trunk = DenseNet(trunk_layer)

    def forward(self, a, grid):
        b = self.branch(a)
        t = self.trunk(grid)
        return torch.einsum('bp,np->bn', b, t)

####################################
TRAIN_PATH = 'piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'piececonst_r421_N1024_smooth2.mat'

s = 85
u0_dim = s ** 2
n_train = 1000
n_test = 100
batch_size = 20

data = scipy.io.loadmat(TRAIN_PATH)
a = data["coeff"][:,::5,::5][:,:49,:49].reshape(1024,-1)
u = data["sol"][:,::5,::5][:,:49,:49].reshape(1024,-1)
a_train = torch.Tensor(a[:n_train]).to(device)
u_train = torch.Tensor(u[:n_train]).to(device)

data = scipy.io.loadmat(TEST_PATH)
a = data["coeff"][:,::5,::5][:,:49,:49].reshape(1024,-1)
u = data["sol"][:,::5,::5][:,:49,:49].reshape(1024,-1)
a_test = torch.Tensor(a[:ntest]).to(device)
u_test = torch.Tensor(u[:ntest]).to(device)
dataloader = DataLoader(torch.utils.data.TensorDataset(a_train, u_train), batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(torch.utils.data.TensorDataset(a_test, u_test), batch_size=batch_size, shuffle=False)

grid = torch2dgrid(s, s)
grid = grid.reshape(-1, 2).to(device)  # grid value, (SxS, d)


branch_layers = [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250]
trunk_layers = [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250]
model = DeepONet(branch_layer=[u0_dim] + branch_layers,
                   trunk_layer=[2] + trunk_layers).to(device)
print(count_params(model))
optimizer = Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)
epochs = range(2000)                    
pbar = epochs
pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

class L2Loss(object):
    """Computes the relative L2 loss between two tensors without size averaging."""
    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p=2, dim=1)
        y_norms = torch.norm(y.reshape(num_examples, -1), p=2, dim=1)
        return torch.sum(diff_norms / y_norms)

myloss = L2Loss()

for e in pbar:
    train_loss = 0.0
    model.train()
    for x, y in dataloader:
        pred = model(x, grid)
        loss = myloss(pred, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= n_train
    scheduler.step()
    
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader_test:
            pred = model(x, grid)
            loss = myloss(pred, y)
            test_loss += loss.item()
        test_loss /= n_test
        scheduler.step()

    pbar.set_description(
        (
            f'Epoch: {e}; Averaged train loss: {train_loss:.5f}; Averaged test loss: {test_loss:.5f}; '
        )
    )