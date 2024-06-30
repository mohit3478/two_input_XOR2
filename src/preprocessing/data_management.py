import os
import pandas as pd
import torch
import torch.nn as nn
from src.config import config

class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

def load_dataset(file_name):
    file_path = os.path.join(config.DATAPATH, file_name)
    data = pd.read_csv(file_path)
    return data

def save_model(model):
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH, "two_input_xor_nn.pth")
    try:
        torch.save(model.state_dict(), pkl_file_path)
        print(f"Saved model at {pkl_file_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")

def load_model(file_name):
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH, file_name)
    model = XORNet()
    model.load_state_dict(torch.load(pkl_file_path))
    model.eval()
    return model
