import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .Models import CNO2d as CNO
from .CRNO import CRNO2d as CRNO
from .DON import DeepONetCP
from .Models import FNO2d as FNO
from .Models import UNet2d as Unet
from .ResNet import ResNet as ResNet
from torch.utils.data import Dataset

import scipy


    
class SinFrequencyDataset(Dataset):
    def __init__(self, which="training", training_samples = 1024, s=64):
        # Note: Normalization constants for both ID and OOD should be used from the training set!
        #Load normalization constants from the TRAINING set:
        file_data_train = "../data/PoissonData_64x64_IN.h5"
        self.reader = h5py.File(file_data_train, 'r')
        self.min_data = self.reader['min_inp'][()]
        self.max_data = self.reader['max_inp'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]
        
        self.file_data = "../data/PoissonData_64x64_IN.h5"

        self.s = s #Sampling rate

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            self.length = 256
            self.start = 1024+128
        
        
        #If the reader changed.
        self.reader = h5py.File(self.file_data, 'r')
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)

        return inputs, labels

   
class NS_High_RDataset(Dataset):  # NS high Reynolds number
    def __init__(self, which="training", training_samples = 1792, s=64):
        with h5py.File("../data/new_ns_train_1e-4_T30_N2048_s64.h5", 'r') as mat_data:
            u_data = np.array(mat_data['u'])  # (50, 64, 64, 10000)
        self.u_data = u_data.transpose(3, 0, 1, 2)  # Reshape to (10000, 50, 64, 64)
        
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = training_samples
        elif which == "test":
            self.length = 128
            self.start = training_samples + 128

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return the sample at index `idx`
        inputs = torch.from_numpy(self.u_data[idx+ self.start,0,...]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.u_data[idx+ self.start,5,...]).type(torch.float32).reshape(1, 64, 64)
        return inputs, labels
        
        
class NSRE5000Dataset(Dataset):  # NS high Reynolds number
    def __init__(self, which="training", training_samples = 768, s=64):
        with h5py.File("../data/ns_test_5e-4_T30_N1024_s64.h5", 'r') as mat_data:
            u_data = np.array(mat_data['u'])  # (1024, 64, 64, 20)
        self.u_data = u_data.transpose(0, 3, 1, 2)
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = training_samples
        elif which == "test":
            self.length = 128
            self.start = training_samples + 128
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return the sample at index `idx`
        inputs = torch.from_numpy(self.u_data[idx+ self.start,0,...]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.u_data[idx+ self.start,10,...]).type(torch.float32).reshape(1, 64, 64)
        return inputs, labels
        

class Darcy_inDataset(Dataset):
    def __init__(self, which="training", training_samples = 800):
        
        self.a = torch.load('../data/synthetic_data_dirichlet_1000_x.pt')
        self.u = torch.load('../data/synthetic_data_dirichlet_1000_y.pt')
        min_val = torch.min(self.u[:training_samples,...])
        max_val = torch.max(self.u[:training_samples,...])
        self.u = (self.u - min_val) / (max_val - min_val)
        min_val = torch.min(self.a[:training_samples,...])
        max_val = torch.max(self.a[:training_samples,...])
        self.a = (self.a - min_val) / (max_val - min_val)
        
        #self.reader = h5py.File(self.file_data, 'r') 
        self.a = self.a[...,::4,::4]
        self.u = self.u[...,::4,::4]
        
        if which == "training":
            self.length = training_samples
            self.start = 0 
        elif which == "validation":
            self.length = 100
            self.start = training_samples
        elif which == "test":
            self.length = 100
            self.start = training_samples+100
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inputs = self.a[idx+ self.start,...].type(torch.float32).reshape(1, 64, 64) 
        labels = self.u[idx+ self.start,...].type(torch.float32).reshape(1, 64, 64)
        return inputs, labels
        

   

class Model_and_Data:
    def __init__(self, which_model, which_example, network_properties, device, batch_size):

        #Must have parameters: ------------------------------------------------        
        if which_model == "CNO":
            in_size = network_properties["in_size"]   
            N_layers = network_properties["N_layers"]
            N_res = network_properties["N_res"]        
            N_res_neck = network_properties["N_res_neck"]        
            channel_multiplier = network_properties["channel_multiplier"]
            out_channel_dim = network_properties["out_channel_dim"]
            ##----------------------------------------------------------------------
            self.model = CNO(in_dim  = 1,      # Number of input channels.
                            size = in_size,                 # Input spatial size
                            N_layers = N_layers,                    # Number of (D) and (U) Blocks in the network
                            N_res = N_res,                          # Number of (R) Blocks per level
                            N_res_neck = N_res_neck,
                            channel_multiplier = channel_multiplier,
                            out_dim=out_channel_dim
                            ).to(device)
        elif which_model == "FNO":
            modes1 = network_properties["modes1"]              
            modes2 = network_properties["modes2"] 
            width = network_properties["hidden_channel_dim"]
            N_layers = network_properties["N_layers"]
            in_channel_dim = network_properties["in_channel_dim"]
            out_channel_dim = network_properties["out_channel_dim"]
            self.model = FNO(modes1=modes1, modes2=modes2, width=width, num_layers=N_layers, in_channel_dim=in_channel_dim, out_channel_dim=out_channel_dim).to(device)   
        elif which_model == "CRNO":
            modes = network_properties["modes"]              
            ini_channels = network_properties["ini_channels"]
            latent_size = network_properties["latent_size"]
            in_out_size = network_properties["in_out_size"]
            N_layers = network_properties["N_layers"]
            N_res = network_properties["N_res"]        
            N_res_neck = network_properties["N_res_neck"]     
            in_channel_dim = network_properties["in_channel_dim"]
            out_channel_dim = network_properties["out_channel_dim"]
            self.model = CRNO(in_dim  = in_channel_dim,      # Number of input channels.
                            out_dim = out_channel_dim,
                            in_out_size = in_out_size,
                            latent_size = latent_size,                 # Latent Spacial size
                            modes = modes,
                            N_layers = N_layers,                    # Number of (D) and (U) Blocks in the network
                            N_res = N_res,                          # Number of (R) Blocks per level
                            N_res_neck = N_res_neck,
                            ini_channel = ini_channels
                            ).to(device)
        elif which_model == "DON":
            branch_layers = network_properties["branch_layers"]              
            trunk_layers = network_properties["trunk_layers"]
            branch_width = network_properties["branch_width"]              
            trunk_width = network_properties["trunk_width"]
        
            branch = [branch_width] * branch_layers
            trunk = [trunk_width] * trunk_layers
            self.model = DeepONetCP(branch_layer=[64*64] + branch,
                               trunk_layer=[2] + trunk).to(device)
        elif which_model == "U":
            in_channel_dim = network_properties["in_channel_dim"]              
            out_channel_dim = network_properties["out_channel_dim"]
            ini_channels = network_properties["ini_channels"]              
            self.model = Unet(in_channel_dim, out_channel_dim, ini_channels).to(device)
        elif which_model == "ResNet":
            # in_channel_dim = network_properties["in_channel_dim"]              
            # out_channel_dim = network_properties["out_channel_dim"]
            # layers = network_properties["layers"]              
            # neurons = network_properties["neurons"]       
            # self.model = ResNet(in_channel_dim, out_channel_dim, layers, neurons).to(device)
            self.model = ResNet().to(device)
        num_workers = 0
        if which_example == "poisson":
            self.train_loader = DataLoader(SinFrequencyDataset("training"), batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.val_loader = DataLoader(SinFrequencyDataset("validation"), batch_size=batch_size, shuffle=False, num_workers=num_workers)
            self.test_loader = DataLoader(SinFrequencyDataset("test"), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        elif which_example == "ns_high_r":
            self.train_loader = DataLoader(NS_High_RDataset("training"), batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.val_loader = DataLoader(NS_High_RDataset("validation"), batch_size=batch_size, shuffle=False, num_workers=num_workers)
            self.test_loader = DataLoader(NS_High_RDataset("test"), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        elif which_example == "ns_re5000":
            self.train_loader = DataLoader(NSRE5000Dataset("training"), batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.val_loader = DataLoader(NSRE5000Dataset("validation"), batch_size=batch_size, shuffle=False, num_workers=num_workers)
            self.test_loader = DataLoader(NSRE5000Dataset("test"), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        elif which_example == "darcy_in":
            self.train_loader = DataLoader(Darcy_inDataset("training"), batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.val_loader = DataLoader(Darcy_inDataset("validation"), batch_size=batch_size, shuffle=False, num_workers=num_workers)
            self.test_loader = DataLoader(Darcy_inDataset("test"), batch_size=batch_size, shuffle=False, num_workers=num_workers)
           
#------------------------------------------------------------------------------

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

