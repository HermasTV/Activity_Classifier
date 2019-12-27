# -*- coding: utf-8 -*-
import torch
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
def data_load(data_folder,transform,train_size=0.6,val_size=0.2,test_size=0.2):
    
    
    dataSet = datasets.ImageFolder(data_folder,transform=transform)
    #identifing the indicies lengths
    train_count = int(len(dataSet) * train_size)
    validation_count = int(len(dataSet) * val_size)
    
    indices = torch.randperm(len(dataSet))
    
    #split train,test and val indicies
    train_indices = indices[0 : train_count]
    #print(len(train_indices))
    validation_indices = indices[train_count : train_count+validation_count]
    #print(len(validation_indices))
    test_indices = indices[train_count + validation_count : train_count + validation_count+len(dataSet)]
    #print(len(test_indices))
    dataloaders = {
    "train": DataLoader(
        dataSet, sampler=SubsetRandomSampler(train_indices), 
    ),
    "val": DataLoader(
        dataSet, sampler=SubsetRandomSampler(validation_indices)
    ),
    "test": DataLoader(
        dataSet, sampler=SubsetRandomSampler(test_indices)
    ),
    }
    return dataloaders



def data_loader(data_folder,transform):
    
    
    dataSet = datasets.ImageFolder(data_folder,transform=transform)
    #identifing the indicies lengths


    dataloader =DataLoader(dataSet,shuffle=True)
    return dataloader