import os
from torch.utils.data import DataLoader
import multiprocessing  # Import for getting the number of CPU cores
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import numpy as np


# CIFAR10 mean and std
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

def get_train_transformers():
    
    train_transforms = A.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        A.HorizontalFlip(p=0.5),
                                        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                                        A.CoarseDropout(
                                            max_holes=1, max_height=16, max_width=16,
                                            min_holes=1, min_height=16, min_width=16,
                                            fill_value=[x * 255.0 for x in mean],
                                            p=0.5,
                                        ),
                                        A.Normalize(mean=mean, std=std),
                                        ToTensorV2()
                                        # transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                        # transforms.ToTensor(),
                                        # transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                        # Note the difference between (0.1307) and (0.1307,)
                                        ])
    return train_transforms
    
    
def get_test_transformers():
    test_transforms = A.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       A.Normalize(mean=mean, std=std),
                                        ToTensorV2()
                                       ])
    return test_transforms

    
def get_CIFAR10_data_loader():
    train_dataset = datasets.CIFAR10('./data', train=True, download=True)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True)

    # Wrap with custom dataset
    train_dataset = CIFAR10Dataset(train_dataset, get_train_transformers())
    test_dataset = CIFAR10Dataset(test_dataset, get_test_transformers())

    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

    return train_loader, test_loader


class CIFAR10Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = np.array(image)
            
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, label
    
    def __len__(self):
        return len(self.dataset)