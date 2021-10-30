import csv, inspect, json
import os.path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import torch
import torch.autograd as autograd
from torchvision import datasets, transforms
from torchvision.transforms.functional import crop
from torch.utils.data import DataLoader, Dataset, random_split

def crop_lemon(image):
    return crop(image, 300, 300, 500, 500)

def load_dataset(dataset, batch_size, test_batch_size, **kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    datadir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../data'
    
    train_data, test_data = None, None

    if dataset.upper() == "MNIST":
        train_data = datasets.MNIST(root=datadir, train=True, transform=transform, download=True)
        test_data = datasets.MNIST(root=datadir, train=False, transform=transform, download=True)
        
    if dataset.upper() == "LEMON":
        train_dir = datadir + "/lemon/train_image_10"
        test_dir = datadir + "/lemon/test_image_10"
        
        train_data = datasets.ImageFolder(train_dir, transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), transforms.Lambda(crop_lemon), transforms.Resize((50,50)), transforms.ToTensor()]))
        test_data = datasets.ImageFolder(test_dir,transforms.Compose([
            transforms.Grayscale(num_output_channels=1), transforms.Lambda(crop_lemon), transforms.Resize((50,50)), transforms.ToTensor()]))
        
        val_size = 100
        train_size = len(train_data) - val_size
        train_data, test_data = random_split(train_data,[train_size,val_size])
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader