import torch
import numpy as np
from src.config import config
from src.preprocessing.data_management import load_model

def predict(input_data):
    model = load_model('two_input_xor_nn.pth')
    with torch.no_grad():
        inputs = torch.tensor(input_data, dtype=torch.float32)
        outputs = model(inputs)
    return outputs.numpy()

if __name__ == "__main__":
    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = predict(test_data)
    print("Predictions:", predictions)
