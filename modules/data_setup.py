import os

import kagglehub
import pandas as pd
from torchvision import datasets, transforms

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

def create_specific_dataset(classes, transform, img_size=224):
    # Possible transforms dictionary
    TRANSFORMS = {
        "Gray": transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]),
        "Color": transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])}
    
    t = TRANSFORMS.get(transform)
    train_dataset = SubsetImageFolder(os.path.join(path, "train"), classes, transform=t)
    val_dataset = SubsetImageFolder(os.path.join(path, "valid"), classes, transform=t)
    test_dataset = SubsetImageFolder(os.path.join(path, "test"), classes, transform=t)

    return train_dataset, val_dataset, test_dataset