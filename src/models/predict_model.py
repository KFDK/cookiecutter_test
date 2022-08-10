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


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("model")
    parser.add_argument("images")
    args = parser.parse_args()

    model = CNN()
    model.load_state_dict(torch.load(args.model))
    model.double()
    data_path = "data/processed/test_dataset.pt"
    test_dataset = torch.load(data_path)
    testloader = DataLoader(test_dataset, batch_size=64)

    with torch.no_grad():
        model.eval()
        running_accs = []
        for images, labels in testloader:
            log_ps = model(images)

            _, top_class = log_ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            running_accs.append(accuracy.item())
    accuracy = sum(running_accs) / len(running_accs)
    print(f"Accuracy: {accuracy*100}%")


if __name__ == "__main__":
    evaluate()
