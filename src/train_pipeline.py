import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.config import config
from src.preprocessing.data_management import load_dataset, save_model, XORNet

def run_training():
    # Ensure the model save directory exists
    if not os.path.exists(config.SAVED_MODEL_PATH):
        os.makedirs(config.SAVED_MODEL_PATH)

    # Load the dataset
    training_data = load_dataset("train.csv")

    # Prepare the data
    X_train = training_data.iloc[:, 0:2].values
    Y_train = training_data.iloc[:, 2].values

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.MINI_BATCH_SIZE, shuffle=True)

    # Define the model
    model = XORNet()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(1000):
        model.train()
        running_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/1000], Loss: {running_loss/len(train_loader):.4f}')

    # Save the model
    print("Saving the model...")
    save_model(model)
    print("Model saved successfully.")

if __name__ == "__main__":
    run_training()
