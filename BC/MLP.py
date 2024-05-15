import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os

from dataset import get_MLP_dataloader, get_MLP_dataloader2, get_pp_dataloader

class BMLPModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(BMLPModel, self).__init__()

        self.input_size = input_size

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.Tanh(),
            nn.Linear(hidden_size1, hidden_size1),
            nn.Tanh(),
            nn.Linear(hidden_size1, hidden_size1),
            nn.Tanh(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Tanh(),
            nn.Linear(hidden_size2, output_size)
        )

        print("Number of parameters: ", sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        return self.network(x)

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLPModel, self).__init__()

        self.input_size = input_size

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.Tanh(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Tanh(),
            nn.Linear(hidden_size2, output_size)
        )

        print("Number of parameters: ", sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        return self.network(x)

class MLPModel2(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPModel2, self).__init__()

        lidar_channels = 1

        self.network = nn.Sequential(
            nn.Conv1d(lidar_channels, 8, kernel_size=11, stride=5), # 1081 - 11 / 5 + 1 = 215
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 8, kernel_size=11, stride=4), # 215 - 11 / 4 + 1 = 52
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Flatten(1),
            nn.Linear(8*52, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.network(x)


def train_model(model, train_loader, criterion, optimizer, lr_scheduler, num_epochs):
    for epoch in range(num_epochs):
        epoch_loss = []  # List to store all losses for the epoch
        for inputs, labels in train_loader:
            if torch.isnan(inputs).any() or torch.isnan(labels).any():
                print("NaN detected in input or labels")
                continue  # Skip this batch if NaN detected
            
            interval = np.ceil(inputs.shape[1] / model.input_size).astype(int)
            inputs = inputs[:, ::interval]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
        
        avg_loss = sum(epoch_loss) / len(epoch_loss)  # Correct variable name used here
        lr_scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.6f}')

        if epoch % 80 == 79:
            save_model(model, f'pp_{model.input_size}_mlp_model_{epoch}.pth')

    save_model(model, f'pp_{model.input_size}_mlp_model_{num_epochs}.pth')

    


def save_model(model, path):
    torch.save(model, path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

if __name__ == "__main__":
    input_size = 1081 # 136 # 271 # 361
    hidden_size1 = 256
    hidden_size2 = 128
    output_size = 2
    num_epochs = 50
    batch_size = 128


    #folder = "./data/Gulf_dataset"
    #train_loader = get_MLP_dataloader2(folder, batch_size, file_num=-1)
    #train_loader = get_MLP_dataloader(folder, batch_size, cat_state_to_obs=False, file_num=-1)

    folder = './data/pp_data'
    train_loader = get_pp_dataloader(folder, batch_size)

    print("Data loaded, starting training, We have {} batches".format(len(train_loader)))

    model = MLPModel(input_size, hidden_size1, hidden_size2, output_size)

    #model.load_state_dict(torch.load('/Users/mac/Desktop/PENN/f1tenth_rl_obs/BC/pp_small_mlp_model_40.pth').state_dict())
    #model = MLPModel2(input_size, output_size)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    train_model(model, train_loader, criterion, optimizer, lr_scheduler, num_epochs)





