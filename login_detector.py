from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import shutil
from pathlib import Path
from random import random
import os
import pandas as pd
import time

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def get_model():
    resnet = torchvision.models.resnet34(pretrained=True)
    
    # Substitute the FC output layer
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 2)
    torch.nn.init.xavier_uniform_(resnet.fc.weight)
    return resnet

def train(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler=None, epochs=10, device='cuda', checkpoint_epochs=10):
    start = time.time()
    print(f'Training for {epochs} epochs on {device}')
    
    for epoch in range(1,epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        
        model.train()  # put network in train mode for Dropout and Batch Normalization
        train_loss = torch.tensor(0., device=device)  # loss and accuracy tensors are on the GPU to avoid data transfers
        train_accuracy = torch.tensor(0., device=device)
        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                train_loss += loss * train_dataloader.batch_size
                train_accuracy += (torch.argmax(preds, dim=1) == y).sum()
        
        if valid_dataloader is not None:
            model.eval()  # put network in train mode for Dropout and Batch Normalization
            valid_loss = torch.tensor(0., device=device)
            valid_accuracy = torch.tensor(0., device=device)
            with torch.no_grad():
                for X, y in valid_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    preds = model(X)
                    loss = criterion(preds, y)

                    valid_loss += loss * valid_dataloader.batch_size
                    valid_accuracy += (torch.argmax(preds, dim=1) == y).sum()
        
        if scheduler is not None: 
            scheduler.step()
            
        print(f'Training loss: {train_loss/len(train_dataloader.dataset):.2f}')
        print(f'Training accuracy: {100*train_accuracy/len(train_dataloader.dataset):.2f}')
        
        if valid_dataloader is not None:
            print(f'Valid loss: {valid_loss/len(valid_dataloader.dataset):.2f}')
            print(f'Valid accuracy: {100*valid_accuracy/len(valid_dataloader.dataset):.2f}')
        
        if epoch%checkpoint_epochs==0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, './checkpoint.pth.tar')
        
        print()
    
    end = time.time()
    print(f'Total training time: {end-start:.1f} seconds')
    return model

def copy_file(source_directory, destination_directory, filename):
    """
    Utility function used to copy a file from a source_directory to a destination_directory
    """
    destination_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_directory/filename, destination_directory/filename)
    
def organize_train_valid_dataset(root, labels, valid_probability=0.1):
    """
    Creates the train, train_valid and valid folders respecting PyTorch's ImageDataset structure, performing
    train/validation split based on the given percentage
    """
    source_directory = root/'original_train'
    
    with os.scandir(source_directory) as it:
        for entry in it:
            if entry.is_file():
                img_index = entry.name.split('.')[0]  # The index is the name of the image except the extension                
                img_class = labels[labels.id==int(img_index)].label.values[0]  # Find the class by looking up the index in the DF
                
                print (f"image class = {img_class}")
                # Randomly assign the image to the valid dataset with probability 'valid_probability'
                channel = Path('train') if random()>valid_probability else Path('valid')
                destination_directory = root/channel/img_class
                
                # Copy the image to either the train or valid folder, and also to the train_valid folder
                copy_file(source_directory, destination_directory, entry.name)
                copy_file(source_directory, root/'train_valid'/img_class, entry.name)

def organize_test_dataset(root):
    """
    Creates the test folder respecting PyTorch's ImageDataset structure, using a dummy 'undefined' label
    """
    source_directory = root/'original_test'
        
    with os.scandir(source_directory) as it:
        for entry in it:
            if entry.is_file():
                img_index = entry.name.split('.')[0]  # The index is the name of the image except the extension

                channel = Path('test')
                destination_directory = root/channel/'undefined'

                copy_file(source_directory, destination_directory, entry.name)


root = Path('/home/mahbub/cs230/project/web_automation/dataset')

# Read in the labels DataFrame with a label for each image
labels = pd.read_csv('/home/mahbub/cs230/project/web_automation/trainLabels.csv')

print(labels)
# Create the train/train_valid/valid folder structure
valid_probability = 0.1
organize_train_valid_dataset(root, labels, valid_probability)

# Create the test folder structure
organize_test_dataset(root)


train_dataset = torchvision.datasets.ImageFolder(
    root/'train', 
    transform=torchvision.transforms.Compose([
        # Resize step is required as we will use a ResNet model, which accepts at leats 224x224 images
        torchvision.transforms.Resize((224,224)),  
        torchvision.transforms.ToTensor(),
    ])
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

means = []
stdevs = []
for X, _ in train_dataloader:
    # Dimensions 0,2,3 are respectively the batch, height and width dimensions
    means.append(X.mean(dim=(0,2,3)))
    stdevs.append(X.std(dim=(0,2,3)))

mean = torch.stack(means, dim=0).mean(dim=0)
stdev = torch.stack(stdevs, dim=0).mean(dim=0)


train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, stdev)
    ])

train_dataset, train_valid_dataset = [
    torchvision.datasets.ImageFolder(folder, transform=train_transforms) for folder in [root/'train', root/'train_valid']]

valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, stdev)
    ])

valid_dataset, test_dataset = [torchvision.datasets.ImageFolder(folder, transform=valid_transforms) for folder in [root/'valid', root/'test']]

num_gpus = torch.cuda.device_count()

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2*num_gpus, pin_memory=True)
train_valid_dataloader = torch.utils.data.DataLoader(train_valid_dataset, batch_size=128, shuffle=True, num_workers=2*num_gpus, pin_memory=True)

valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=2*num_gpus, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2*num_gpus, pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr, weight_decay, epochs = 1e-5, 5e-4, 15

model = get_model().to(device)

# Standard CrossEntropy Loss for multi-class classification problems
criterion = torch.nn.CrossEntropyLoss()

# params_1x are the parameters of the network body, i.e., of all layers except the FC layers
params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
optimizer = torch.optim.Adam([{'params':params_1x}, {'params': model.fc.parameters(), 'lr': lr*10}], lr=lr, weight_decay=weight_decay)


trained_model = train(model, train_dataloader, valid_dataloader, criterion, optimizer, None, epochs, device)


preds = []

trained_model.eval()
with torch.no_grad():
    for X, _ in test_dataloader:
        X = X.to(device)
        preds.extend(model(X).argmax(dim=1).type(torch.int32).cpu().numpy())

ids = list(range(1, len(test_dataset)+1))
ids.sort(key=lambda x: str(x))

df = pd.DataFrame({'id': ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_dataset.classes[x])
df.to_csv('inference.csv', index=False)