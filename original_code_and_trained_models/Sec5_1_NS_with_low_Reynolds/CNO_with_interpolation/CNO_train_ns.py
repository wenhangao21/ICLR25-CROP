import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer
import math
import random
from CNO import CNO
# from ipdb import set_trace
import argparse
import os
folder = "CNO"
os.makedirs(folder, exist_ok=True)

parser = argparse.ArgumentParser(description='Train models for various examples.')
parser.add_argument('--seed', type=int, required=True, help="Specify the random seed (e.g., 0)")
args = parser.parse_args()
random_seed = args.seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

TRAIN_PATH = 'your_data_path'
TEST_PATH = 'your_data_path'

ntrain = 1000
ntest = 200

modes = 12
width = 20

batch_size = 5
learning_rate = 0.001
epochs = 500
iterations = epochs*(ntrain//20)

path = 'ns_fourier_2d_time_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

sub = 1
S = 64
T_in = 10
T = 40 # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;
step = 1

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)

# set_trace()

train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in].to(device)
train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in].to(device)

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in].to(device)
test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in].to(device)


print(train_u.shape)
print(test_u.shape)
# assert (S == train_u.shape[-2])
# assert (T == train_u.shape[-1])
# set_trace()
train_a = train_a.permute([0,3,1,2])
train_u = train_u.permute([0,3,1,2])
test_a = test_a.permute([0,3,1,2])
test_u = test_u.permute([0,3,1,2])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False, drop_last=True)

################################################################
# training and evaluation
################################################################
# model = FNO2d(modes, modes, width).cuda()
model = CNO(in_size = 64, N_layers = 3, in_dim  = 10).cuda()
print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
accumulation_steps = 4
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    optimizer.zero_grad()  # Initialize optimizer gradient

    for batch_idx, (xx, yy) in enumerate(train_loader):
        loss = 0

        for t in range(0, T, step):
            y = yy[:, t:t + step, :, :]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), 1)

            xx = torch.cat((xx[:, step:, :, :], im), dim=1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        # Backpropagation with gradient accumulation
        loss = loss   # Normalize loss by accumulation steps
        loss.backward()

        # Update optimizer only after the specified accumulation steps
        if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()  # Reset gradients

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0

            for t in range(0, T, step):
                y = yy[:, t:t + step, :, :]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), 1)

                xx = torch.cat((xx[:, step:, :, :], im), dim=1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    if ep % 20 == 0 or ep == epochs - 1:
        print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
              test_l2_full / ntest)
# torch.save(model, path_model)

torch.save(model.state_dict(), f'CNO/CNO_{random_seed}.pth')