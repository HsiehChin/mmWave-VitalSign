import os, yaml
import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# LSTM model
class LSTM(nn.Module):
    def __init__(self, opts):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size = opts.input_size,
                                  hidden_size = opts.hidden_size, 
                                  num_layers = opts.num_layers,
                                  batch_first = True,
                                  bidirectional = opts.isBidirectional,
                                  dropout = opts.dropout)
        
        for name, param in self.lstm.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0.0)
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        self.out = torch.nn.Linear(opts.hidden_size*2, opts.output_size)
        
    def forward(self, sequence): # multi-to-one
        r_out,_ = self.lstm(sequence)
        out = self.out(r_out)

        return out
        

# LSTM fitness model
class LSTM_fitness(nn.Module):
    def __init__(self, opts):
        super(LSTM_fitness, self).__init__()

        self.c_hidden_size = 128
        self.contextual = torch.nn.GRU(input_size = 4, # power+distance+ipaq
                                    hidden_size = self.c_hidden_size, 
                                    num_layers = 1,
                                    batch_first = True,
                                    bidirectional = opts.isBidirectional)

        self.c_fc = torch.nn.Linear(self.c_hidden_size*2, opts.output_size)

        self.lstm = torch.nn.LSTM(input_size = opts.input_size,
                                  hidden_size = opts.hidden_size, 
                                  num_layers = opts.num_layers,
                                  batch_first = True,
                                  bidirectional = opts.isBidirectional,
                                  dropout = opts.dropout)

        for name, param in self.contextual.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0.0)
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        for name, param in self.lstm.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0.0)
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        self.out = torch.nn.Linear(opts.hidden_size*2, opts.output_size)

    def zscore_normalization(self, sequence):
        mean_values = torch.mean(sequence, dim=0)
        std_values = torch.std(sequence, dim=0)

        # feature Z-Score normalization
        normalized_data = (sequence - mean_values) / std_values

        return normalized_data        

    def forward(self, sequence):
        power = sequence[:, :, 0].unsqueeze(2)
        distance = sequence[:, :, 1].unsqueeze(2)
        move = sequence[:, :, 2].unsqueeze(2)
        ipaq = sequence[:, :, 3:]
        
        show = {} # for debug

        # contextual layer
        c_input = torch.cat((power, distance, ipaq), dim=2)  
        c_out, _ = self.contextual(c_input)     
        c_out = self.c_fc(c_out)
        show['contextual output'] = c_out

        c_out = torch.abs(c_out)
        show['abs'] = c_out

        c_out = self.zscore_normalization(c_out)
        show['z-score Normalization'] = c_out
        
        # move curve feature, c_output, move, ipaq
        l2_input = torch.cat((c_out, move, ipaq), dim=2)
        r2_out, _ = self.lstm(l2_input)
        out = self.out(r2_out)

        return out, show



def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def MAELoss(yhat,y):
    mae = nn.L1Loss()
    m = mae(yhat, y)
    return m
# The current loss function
def MSELoss(yhat,y):
    mse = nn.MSELoss()
    m2 = mse(yhat, y)
    return m2

# Using Loss type in config to decide the loss function.
def vital_loss_func(LOSS_TYPE):
    if LOSS_TYPE=="MAE":
        return MAELoss
    if LOSS_TYPE=="MSE":
        return MSELoss
    if LOSS_TYPE=="RMSE":
        return RMSELoss

# Read config file to get LSTM model parameters 
def get_lstm_parameters(yaml_path):
    config_path = os.path.abspath(os.path.join(os.getcwd(),yaml_path))
    with open(config_path) as stream:
        try:
            config = yaml.full_load(stream)
            opts = EasyDict(config['lstm'])
            return opts
            
        except yaml.YAMLError as exc:
            print(exc)
            return {}

# Get LSTM model 
def lstm_model(yaml_path):
    opts = get_lstm_parameters(yaml_path)
    if opts:
        model = LSTM(opts)
        return model, opts

# Get LSTM fitness model 
def lstm_fitness_model(yaml_path): 
    opts = get_lstm_parameters(yaml_path)
    if opts:
        model = LSTM_fitness(opts)
        return model, opts