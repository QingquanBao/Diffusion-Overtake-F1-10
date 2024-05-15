import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import re
import gc
import random

from torch.utils.data import TensorDataset, DataLoader, Dataset


STATE_IDX_MAP = {
    'x': 0,
    'y': 1,
    'theta': 2,
    'vx': 3,
    'vy': 4,
    'steering': 5,
    'angular_speed': 6,
    'slip_angle': 7,
}


def loading_data(folder):
    check_name = re.compile('^ML')

    datasets = []
    for filename in sorted(os.listdir(folder)):
        files = os.path.join(folder, filename)
        if re.match(check_name, filename) and os.path.isfile(files):
            temp = pd.read_csv(files, skiprows=1, header=None).iloc[:, :-1]
            temp[0] = temp[0].map(lambda t : t/16.0)
            temp[1] = temp[1].map(lambda t : t/0.192)
            temp = temp.to_numpy()
            temp[:, 2:] = temp[:, 2:] / 10.0
            datasets.append(np.array(temp, dtype=float))
    gc.collect()

    return datasets

def loading_vehicle_state_data(folder):
    check_name = re.compile('^car_state_blue')

    datasets = []
    for filename in sorted(os.listdir(folder)):
        files = os.path.join(folder, filename)
        if re.match(check_name, filename) and os.path.isfile(files):
            #datasets.append(np.array(pd.read_csv(files).iloc[:, [3, 5]], dtype=float))
            datasets.append(np.array(pd.read_csv(files), dtype=float))

    gc.collect()
    return datasets


def get_organized_data(folder, file_num=-1):
    """
    return the scan, cmd, and state_list
    scan: list of tensor, each tensor is (T, scan_dim) 
    cmd: list of tensor, each tensor is (T, 2)
    state_list: list of tensor, each tensor is (T, 2)
    """

    # Read scan and command data
    # scan is the observation, cmd is the desired output
    datasets = loading_data(folder)
    states = loading_vehicle_state_data(folder)

    scan = []
    cmd = []
    for x in datasets:
        scan.append(x[:, 2:])
        cmd.append(x[:, 0:2])

    scan_1 = [torch.tensor(arr) for arr in scan[:file_num]]
    cmd_1 = [torch.tensor(arr) for arr in cmd[:file_num]]
    state_list = [torch.tensor(arr) for arr in states[:file_num]]

    datasets = []
    scan = []
    cmd = []
    states = []
    gc.collect()

    return scan_1, cmd_1, state_list


def get_data_loader(folder, file_num=-1, special_value=-100.0):

    scan_1, cmd_1, state_list = get_organized_data(folder, file_num)

    scanpad_A = torch.nn.utils.rnn.pad_sequence(scan_1, 
                                                batch_first=True, 
                                                padding_value=special_value)
    cmdpad = torch.nn.utils.rnn.pad_sequence(cmd_1, 
                                             batch_first=True, 
                                             padding_value=special_value)
    # Read vehicle state
    statespad_B = torch.nn.utils.rnn.pad_sequence(state_list, 
                                                  batch_first=True, 
                                                  padding_value=special_value)


    Xpad_Atrain_tensor = torch.tensor(scanpad_A, dtype=torch.float32)
    Xpad_Btrain_tensor = torch.tensor(statespad_B, dtype=torch.float32)
    cmdpad_train_tensor = torch.tensor(cmdpad, dtype=torch.float32)

    dataset = TensorDataset(Xpad_Atrain_tensor, Xpad_Btrain_tensor, cmdpad_train_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return train_loader

def get_MLP_dataloader(folder, batch_size=32, cat_state_to_obs=False, file_num=-1):
    """
    return the dataloader for the MLP model
    data would be looks like this:
        obs: (B, obs_dim)
        action: (B, action_dim) 
    """
    scan_1, cmd_1, state_list = get_organized_data(folder, file_num)

    # Only use vx and steering
    state_list = [x[:, [STATE_IDX_MAP['vx'], STATE_IDX_MAP['steering']]] for x in state_list]

    scan = torch.cat(scan_1, dim=0)
    cmd = torch.cat(cmd_1, dim=0)
    state = torch.cat(state_list, dim=0)
    if cat_state_to_obs:
        obs = torch.cat([scan, state], dim=1)
    else:
        obs = scan

    dataset = TensorDataset(obs.float(), cmd.float())
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def get_MLP_dataloader2(folder, batch_size=32, file_num=-1):
    """
    return the dataloader for the MLP model
    data would be looks like this:
        obs: (B, obs_dim)
        action: (B, action_dim) 
    """
    scan_1, cmd_1, state_list = get_organized_data(folder, file_num)
    scan = torch.cat(scan_1, dim=0)

    # use state vx and steering as the command
    #cmd = [x[:, [STATE_IDX_MAP['vx'], STATE_IDX_MAP['steering']]] for x in state_list]
    velocity = [ (x[:, STATE_IDX_MAP['vx']] **2 + x[:, STATE_IDX_MAP['vy']] **2) ** 0.5 for x in state_list]
    steering = [x[:, STATE_IDX_MAP['steering']] for x in state_list]

    cmd = torch.stack([
                        torch.cat(steering, dim=0) / 0.192, 
                        torch.cat(velocity, dim=0) / 10.0,
                        ],
                        dim=1)

    obs = scan

    dataset = TensorDataset(obs.float(), cmd.float())
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def get_pp_dataloader(folder, batch_size=32):
    obs = np.load(os.path.join(folder, 'obs.npy')) / 10
    action = np.load(os.path.join(folder, 'action.npy')) / np.array([0.192, 10])

    obs = torch.tensor(obs, dtype=torch.float32)
    action = torch.tensor(action, dtype=torch.float32)

    dataset = TensorDataset(obs, action)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

class DiffusionDataset(Dataset):
    def __init__(self, 
                 folder, 
                 obs_horizon=10,
                action_horizon=10,
                 file_num=-1):
        self.file_num = file_num
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        #self._read_data(folder)
        self._read_sim_data(folder)

    def _read_sim_data(self, folder):
        obs = np.load(os.path.join(folder, 'obs.npy')) / 10
        action = np.load(os.path.join(folder, 'action.npy')) / np.array([0.192, 10])

        self.scan = torch.tensor(obs, dtype=torch.float32)
        self.cmd = torch.tensor(action, dtype=torch.float32)

        self.file_len = len(self.scan)
        self.file_len_dict = {0: len(self.scan)}


    def _read_data(self, folder):
        datasets = loading_data(folder)
        speed_steering = loading_vehicle_state_data(folder)

        scan = []
        cmd = []
        for x in datasets:
            scan.append(x[:, 2:])
            cmd.append(x[:, 0:2])

        scan_1 = scan[:self.file_num]
        cmd_1 = cmd[:self.file_num]

        self.scan = [torch.tensor(arr) for arr in scan_1[:self.file_num]]
        self.state = [torch.tensor(arr) for arr in speed_steering[:self.file_num]]
        self.cmd = [torch.tensor(arr) for arr in cmd_1[:self.file_num]]

        datasets = []
        scan = []
        cmd = []
        speed_steering = []
        gc.collect()

        self.file_len = len(self.scan)
        self.file_len_dict = {i: len(x) for i, x in enumerate(self.scan)}
    
    def __len__(self):
        full_len = sum([x.shape[0] for x in self.scan]) - self.file_len * (self.obs_horizon + self.action_horizon - 1)
        return full_len

    def __getitem__(self, idx):
        # get the file number
        file_idx = 0
        while idx >= self.scan[file_idx].shape[0] - self.obs_horizon - self.action_horizon + 1:
            idx -= self.scan[file_idx].shape[0] - self.obs_horizon - self.action_horizon + 1
            file_idx += 1

        # get the idx in the file
        idx += self.obs_horizon

        obs = self.scan[file_idx][idx - self.obs_horizon:idx]
        #state = self.state[file_idx][idx - self.obs_horizon:idx]
        action = self.cmd[file_idx][idx:idx + self.action_horizon]

        return {'obs': obs.float(), 'action':action.float()}


def get_Diffusion_dataloader(folder, 
                            obs_horizon=10,
                            action_horizon=10,
                            batch_size=32,
                             file_num=-1):
    """
    return the dataloader for the diffusion model
    data would be looks like this:
        obs: (B, obs_horizon, obs_dim)
        action: (B, action_horizon, action_dim) 

    Here obs_dim = scan_dim + state_dim
         action_dim = 2
    """

    dataset = DiffusionDataset(folder, 
                            obs_horizon,
                            action_horizon,
                            file_num)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader

if __name__ == '__main__':
    folder = "data/Gulf_dataset"
    train_loader = get_data_loader(folder, 3)
    print(train_loader)