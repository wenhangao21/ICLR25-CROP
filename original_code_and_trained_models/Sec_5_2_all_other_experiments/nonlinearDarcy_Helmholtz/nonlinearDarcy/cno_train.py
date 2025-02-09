import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
import random
import argparse
from CNO import CNO2d
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
# configs
################################################################
TRAIN_PATH = 'piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'piececonst_r421_N1024_smooth2.mat'

ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001
epochs = 500
iterations = epochs*(ntrain//batch_size)

modes = 12
width = 32

r = 5
h = int(((421 - 1)/r) + 1)
s = h

################################################################
# load data and data normalization
################################################################
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s].cuda()
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s].cuda()

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s].cuda()
y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s].cuda()

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

print(x_train.shape)
x_train = x_train.reshape(ntrain,s,s,1)
x_test = x_test.reshape(ntest,s,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
model = CNO2d(in_dim  = 1, size = s, 
                            N_layers = 3,                    # Number of (D) and (U) Blocks in the network
                            N_res = 4,                          # Number of (R) Blocks per level
                            N_res_neck = 6,
                            channel_multiplier = 32,
                            out_dim = 1
                            ).cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s, s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_l2 += loss.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x).reshape(batch_size, s, s)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_l2/= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    if ep % 20 == 0 or ep == epochs -1:
      print(ep, t2-t1, train_l2, test_l2)
    with open(f"models/FNO_{random_seed}.txt", "a") as f:
        if ep % 20 == 0 or ep == epochs - 1:
            log_message = (f"{ep}, {t2-t1}, {train_l2}, {test_l2}\n")
            f.write(log_message)  
            
torch.save(model.state_dict(), f"models/FNO_{random_seed}.pth")    