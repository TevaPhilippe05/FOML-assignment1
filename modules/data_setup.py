import os

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torchvision import datasets, transforms

# Load the dataset and dataframe
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(MODULE_DIR), 'output')

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

def class_distribution(df, sports_list, figsize=(8, 4), save_path="dataset_distribution.png"):
    """
    Visualize class distribution for selected sports.
    
    Parameters:
    - df: DataFrame with 'labels' column
    - sports_list: List of sports to filter
    - figsize: Figure size (default: (8, 4))
    - palette: Color palette (default: 'inferno')
    - save_path: Path to save the figure (default: "dataset_distribution.png")
    """
    
    # Filter dataframe
    filtered_df = df[df['labels'].isin(sports_list)]
    
    print(filtered_df["labels"].unique())
    print(f"\nOriginal: {len(df)} rows")
    print(f"Filtered: {len(filtered_df)} rows")
    print(filtered_df.columns)
    print(filtered_df.head())

    # Calculate value counts for 'data set' column
    print("\n--- Data Set Distribution ---")
    dataset_counts = filtered_df['data set'].value_counts()
    print(dataset_counts)
    
    # Visualize class distribution
    filtered_counts = filtered_df["labels"].value_counts()
    plt.figure(figsize=figsize)
    sns.barplot(y=filtered_counts.index, x=filtered_counts.values, hue=filtered_counts.index, 
                palette="inferno", legend=False)
    plt.title('Amount of images per label', fontsize=12)
    plt.ylabel('Labels', fontsize=10)
    plt.xlabel('Count', fontsize=10)
    
    # Add value labels on bars for clarity
    for i, v in enumerate(filtered_counts.values):
        plt.text(v + 0.5, i, str(v), va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save using absolute path
    save_file = os.path.join(OUTPUT_DIR, save_path)
    plt.savefig(save_file, dpi=150)
    print(f"Figure saved to: {save_file}")
    plt.show()