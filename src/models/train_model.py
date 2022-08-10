import argparse
import pdb
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.models.model import CNN


def train():
    print("Training day and night")

    model = CNN()
    data_path = "data/processed/train_dataset.pt"
    train_dataset = torch.load(data_path)
    trainloader = DataLoader(train_dataset, batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    steps = 0

    epochs = 5
    criterion = nn.NLLLoss()

    train_losses = []

    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            # images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            train_losses.append(running_loss / len(trainloader))
            print(f"Training loss: {running_loss/len(trainloader)}")

    # save model
    torch.save(model.state_dict(), "models/trained_model.pth")

    # plotting
    plt.plot(np.arange(0, epochs), train_losses)
    plt.show()
    plt.savefig("reports/figures/train_loss_vs_epoch.png")


if __name__ == "__main__":
    train()
