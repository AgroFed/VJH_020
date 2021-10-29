import csv, inspect, json
import os.path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import torch
import torch.autograd as autograd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

def load_dataset(dataset, batch_size, **kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    datadir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../data'
    
    train_data, test_data = None, None

    if dataset.upper() == "MNIST":
        train_data = datasets.MNIST(root=datadir, train=True, transform=transform, download=True)
        test_data = datasets.MNIST(root=datadir, train=False, transform=transform, download=True)
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader