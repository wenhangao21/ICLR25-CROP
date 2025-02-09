import copy
import json
import os
import sys

import pandas as pd

import torch
import random
import numpy as np

from tqdm import tqdm


from scr.PDE_Examples import Model_and_Data
from scr.utils import LpLoss
from scr.DON import torch2dgrid


#-----------------------------------Setup--------------------------------------
if len(sys.argv) == 4:
    
    training_properties = {
        "learning_rate": 0.001, 
        "weight_decay": 1e-6,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 1000,
        "batch_size": 16,
        "lp": 2,                # Do we use L1 or L2 errors? Default: L1
    }
    
    example_list = ["poisson", "ns_high_r", "ns_re5000", "darcy_in"]
    which_example = sys.argv[1]
    if not which_example in example_list:
        raise ValueError("Command-line arguments: which_example not valid")
    # if which_example == "ns_high_r":
        # training_properties["batch_size"] = 100
    which_model = sys.argv[2]
    if which_model == "CNO":
        model_architecture_ = {
        
        #Parameters to be chosen with model selection:
        "N_layers": 3,            # Number of (D) & (U) blocks 
        "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
        "N_res": 4,               # Number of (R) blocks in the middle networs.
        "N_res_neck" : 6,         # Number of (R) blocks in the BN
        
        #Other parameters:
        "in_size": 64,            # Resolution of the computational grid
        "kernel_size": 3,         # Kernel size
        "in_channel_dim": 1,       # Number of input channels
        "out_channel_dim": 1      # Number of output channels  
        }
        if which_example == "darcy421":
              model_architecture_["in_size"] = 106
    elif which_model == "FNO":
        model_architecture_ = {
        
        #Parameters to be chosen with model selection:
        "modes1": 24,              # Number of Fourier modes for axis 1
        "modes2": 24,              # Number of Fourier modes for axis 2
        "hidden_channel_dim": 16, # Number of channels in intermidiate Fourier layers, also called width
        "N_layers": 4,            # Number of Fourier layers
        
        #Other parameters:
        "in_channel_dim": 1,       # Number of input channels
        "out_channel_dim": 1      # Number of output channels  
        }
    elif which_model == "CRNO":
        model_architecture_ = {
        
        #Parameters to be chosen with model selection:
        "modes": 24,              # Number of Fourier modes for axis 1
        "ini_channels": 32,       # Number of channels in intermidiate Fourier layers, also called width
        "N_layers": 3,            # Number of Fourier layers
        "N_res": 4,               # Number of (R) blocks in the middle networs.
        "N_res_neck" : 6,         # Number of (R) blocks in the BN
        
        #Other parameters:
        "in_out_size": 64,            # Latent Grid Size
        "latent_size": 64,            # Latent Grid Size
        "kernel_size": 3,         # Kernel size
        "in_channel_dim": 1,       # Number of input channels
        "out_channel_dim": 1      # Number of output channels  
        }   
        if which_example == "darcy421":
              model_architecture_["in_out_size"] = 106
              model_architecture_["latent_size"] = 106
    elif which_model == "U":
        model_architecture_ = {
        "in_channel_dim": 1,       # Number of input channels
        "out_channel_dim": 1,      # Number of output channels  
        "ini_channels": 32
        }   
    elif which_model == "ResNet":
        training_properties = {
        "learning_rate": 0.0005,
        "weight_decay": 1e-12,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 1000,
        "batch_size": 16,
        "lp": 2,
        }
        model_architecture_ = {
        "in_channel_dim": 1,       # Number of input channels
        "out_channel_dim": 1,      # Number of output channels  
        "layers": 4,
        "neurons": 40
        }
    else:
        raise ValueError("which_model can be: FNO, CNO, UNet, ResNet, or CRNO")
    random_seed = sys.argv[3]

    # Save the models here:
    folder = "TrainedModels/"+which_model+"_" +which_example+"_"+"seed_"+random_seed
        
else:
    raise ValueError("Command-line arguments: python script name (TrainCRNO.py), which_example, which_model, random_seed")


torch.manual_seed(int(random_seed))
np.random.seed(int(random_seed))
random.seed(int(random_seed))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if which_model == "DON":
    grid = torch2dgrid(64, 64, bot=(0,0), top=(1,1))
    grid = grid.reshape(-1, 2).to(device)  # grid value, (SxS, d)
        
learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
p = training_properties["lp"]

if not os.path.isdir("TrainedModels"):
    print("Generated new folder TrainedModels")
    os.mkdir("TrainedModels")

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

example = Model_and_Data(which_model, which_example, model_architecture_, device, batch_size)
#-----------------------------------Train--------------------------------------
model = example.model
n_params = model.print_size()
train_loader = example.train_loader #TRAIN LOADER
test_loader = example.test_loader #TEST LOADER
val_loader = example.val_loader   #VAL LOADER

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

if p == 1:
    loss = torch.nn.L1Loss()
    loss_lp = LpLoss(p=1, size_average=True)
elif p == 2:
    loss = torch.nn.MSELoss() # MSE and l2 optimization are the same 
    loss_lp = LpLoss(p=2, size_average=True)
    

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")


best_val_loss = float('inf')  # Initialize to a large number
best_test_loss = float('inf')  # Initialize for tracking best test loss
patience = 100  # Number of epochs with no improvement before stopping
counter = 0  # Initialize early stopping counter

for epoch in range(epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        
        # Training loop
        for step, (input_batch, output_batch) in enumerate(train_loader):
            if which_model == "DON":
                N = input_batch.shape[0]
                input_batch= input_batch.reshape(N, -1)
                output_batch = output_batch.reshape(N, -1)
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            if which_model == "DON":
                output_pred_batch = model(input_batch, grid)
            else:
                output_pred_batch = model(input_batch)

            loss_f = loss(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch)
            loss_f.backward()
            optimizer.step()
            
            train_mse = train_mse * step / (step + 1) + loss_f.item() / (step + 1)
            tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse})
            tepoch.update(1)

        # Validation and evaluation
        with torch.no_grad():
            model.eval()

            # Test/Validation loader loss calculation
            test_relative_l2 = 0.0
            for step, (input_batch, output_batch) in enumerate(test_loader):
                if which_model == "DON":
                    N = input_batch.shape[0]
                    input_batch= input_batch.reshape(N, -1)
                    output_batch = output_batch.reshape(N, -1)
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                if which_model == "DON":
                    output_pred_batch = model(input_batch, grid)
                else:
                    output_pred_batch = model(input_batch)

                loss_f = loss_lp(output_pred_batch, output_batch) * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(test_loader)

            # Validation loader loss calculation
            val_loss = 0.0
            for step, (input_batch, output_batch) in enumerate(val_loader):
                if which_model == "DON":
                    N = input_batch.shape[0]
                    input_batch= input_batch.reshape(N, -1)
                    output_batch = output_batch.reshape(N, -1)
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                if which_model == "DON":
                    output_pred_batch = model(input_batch, grid)
                else:
                    output_pred_batch = model(input_batch)

                loss_f = loss_lp(output_pred_batch, output_batch) * 100
                val_loss += loss_f.item()
            val_loss /= len(val_loader)

            # Training loader loss calculation
            train_relative_l2 = 0.0
            for step, (input_batch, output_batch) in enumerate(train_loader):
                if which_model == "DON":
                    N = input_batch.shape[0]
                    input_batch= input_batch.reshape(N, -1)
                    output_batch = output_batch.reshape(N, -1)
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                if which_model == "DON":
                    output_pred_batch = model(input_batch, grid)
                else:
                    output_pred_batch = model(input_batch)

                loss_f = loss_lp(output_pred_batch, output_batch) * 100
                train_relative_l2 += loss_f.item()
            train_relative_l2 /= len(train_loader)

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_relative_l2  # Save corresponding test loss
                best_model = copy.deepcopy(model)
                torch.save(best_model.state_dict(), folder + "/best_model.pth")
                counter = 0  # Reset counter if we have an improvement
            else:
                counter += 1  # Increment counter if no improvement

        # Update progress bar
        tepoch.set_postfix({
            'Train loss': train_mse, 
            "Relative Train": train_relative_l2, 
            "Relative Test": test_relative_l2,
            "Relative Val": val_loss
        })
        tepoch.close()

        # Write current results to file
        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training MSE: " + str(train_mse) + "\n")
            file.write("Testing Relative l2: " + str(test_relative_l2) + "\n")
            file.write("Validation Loss: " + str(val_loss) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")
        
        # Learning rate scheduler step
        scheduler.step()

        # Record errors every 20 epochs
        if (epoch + 1) % 20 == 0:
            with open(folder + '/record_errors.txt', 'a') as file:
                file.write(f"Epoch {epoch}: Train MSE {train_mse}, Train L2 {train_relative_l2}, Test L2 {test_relative_l2}, Validation Loss {val_loss}\n")

        # Early stopping check
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.4f} with test loss: {best_test_loss:.4f}")
            break  # Stop training if patience exceeded

# Save best test loss after training
with open(folder + '/best_test_loss.txt', 'w') as file:
    file.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
    file.write(f"Corresponding Test Loss: {best_test_loss:.4f}\n")