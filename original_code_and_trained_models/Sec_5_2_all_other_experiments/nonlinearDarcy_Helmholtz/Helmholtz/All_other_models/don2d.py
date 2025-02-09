import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from timeit import default_timer
from utilities3 import *
from DON import DeepONetCP as DON2d
from DON import torch2dgrid
from Adam import Adam

import random
import argparse
################################################################
# fourier layer
################################################################
parser = argparse.ArgumentParser(description='Train models for various examples.')
parser.add_argument('--random_seed', type=int, required=True, help="Specify the random seed (e.g., 0)")
args = parser.parse_args()
random_seed = args.random_seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


################################################################
# fourier layer
################################################################

################################################################
# configs
################################################################
ntrain = 800
ntest = 200

batch_size = 20
learning_rate = 0.001

epochs = 2000
step_size = 100
gamma = 0.5

modes = 12
width = 32

# r = 5
r = 1
# s = h
s = 101//r
s1 = s
s2 = s


inputs = np.load("Helmholtz_inputs_normalized_1000.npy")
outputs = np.load("Helmholtz_outputs_normalized_1000.npy")
x_train = inputs[:,:,:ntrain].transpose(2,0,1)
x_test = inputs[:,:,-ntest:].transpose(2,0,1)

y_train = outputs[:,:,:ntrain].transpose(2,0,1)
y_test = outputs[:,:,-ntest:].transpose(2,0,1)

x_train = x_train.reshape(ntrain,1, s1,s2)
x_test = x_test.reshape(ntest,1, s1,s2)

y_train = y_train.reshape(ntrain,1,s1,s2)
y_test = y_test.reshape(ntest,1,s1,s2)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = torch.from_numpy(x_train).to(torch.float32)
x_test = torch.from_numpy(x_test).to(torch.float32)

y_train = torch.from_numpy(y_train).to(torch.float32)
y_test = torch.from_numpy(y_test).to(torch.float32)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=True)

################################################################
# training and evaluation
################################################################
branch_layers = 12              # Number of Fourier modes for axis 1
trunk_layers = 12
branch_width = 250
trunk_width = 250
branch = [branch_width] * branch_layers
trunk = [trunk_width] * trunk_layers
model = DON2d(branch_layer=[101*101]+ branch, trunk_layer=[2] + trunk).cuda()
grid = torch2dgrid(101, 101, bot=(0,0), top=(1,1))
grid = grid.reshape(-1, 2).to(device)  # grid value, (SxS, d)
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
# y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x, grid).reshape(batch_size, s1, s2)

        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x,grid).reshape(batch_size, s1, s2)

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_l2/= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    if ep % 20 == 0 or ep == epochs -1:
      print(ep, t2-t1, train_l2, test_l2)
    with open(f"DON/DON_{random_seed}.txt", "a") as f:
        if ep % 20 == 0 or ep == epochs - 1:
            log_message = (f"{ep}, {t2-t1}, {train_l2}, {test_l2}\n")
            f.write(log_message)  
            
torch.save(model.state_dict(), f"DON/DON_{random_seed}.pth")    