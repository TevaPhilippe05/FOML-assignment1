import os
import torch
import shutil
import kagglehub
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from sklearn.metrics import confusion_matrix

# Set image size
IMG_SIZE = 224

# Load the dataset and dataframe
path = kagglehub.dataset_download("gpiosenka/sports-classification")
sports_dataframe = pd.read_csv(os.path.join(path, "sports.csv"))

# A generated function to extract only given classes
class SubsetImageFolder(datasets.ImageFolder):
    def __init__(self, root, chosen_classes, transform=None):
        self.chosen_classes = chosen_classes
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        classes = sorted(self.chosen_classes)
        class_to_idx = {}
        for i in range(len(classes)):
            class_to_idx[classes[i]] = i
        return classes, class_to_idx

# Possible transforms dictionary
TRANSFORMS = {
    "Gray": transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor(),
]),
    "Color": transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])}

def create_specific_dataset(classes, transform):
    t = TRANSFORMS.get(transform)
    train_dataset = SubsetImageFolder(os.path.join(path, "train"), classes, transform=t)
    val_dataset = SubsetImageFolder(os.path.join(path, "valid"), classes, transform=t)
    test_dataset = SubsetImageFolder(os.path.join(path, "test"), classes, transform=t)

    return train_dataset, val_dataset, test_dataset