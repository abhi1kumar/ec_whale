#!/usr/bin/python3

import os
import argparse
import glob
import numpy as np
import random
from datetime import datetime

import sys
sys.path.insert(0, './src')
from CustomDatasetFromCSV      import CustomDatasetFromCSV
from CustomDatasetFromCSVTrain import CustomDatasetFromCSVTrain

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchsummary import summary

########################################################################################
# Code taken from
# https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983
# https://discuss.pytorch.org/t/converting-2d-to-1d-module/25796/3
########################################################################################
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x): 
        return x.view(x.size(0), -1)


########################################################################################
# Scoring
# https://github.com/pytorch/examples/blob/master/mnist/main.py
########################################################################################
def get_score_model(model, data_loader):
    #toggle model to eval mode
    model.eval()

    test_loss = 0
    correct = 0
    
    #turn off gradients since they will not be used here
    # this is to make the inference faster
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.cross_entropy_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    #test_loss /= len(data_loader.dataset)
    score = correct/float(len(data_loader.dataset))    
    return score


########################################################################################
# Training function
########################################################################################
def train_epoch(model, criterion, optimizer, device, train_loader):   
    losses = AverageMeter() 
       
    ########################################################################################
    # Start Training
    ########################################################################################
    # Transfer model and loss to the default device
    model     = model.to(device)
    criterion = criterion.to(device)
                
    # model for training
    model.train()
    train_losses = []

    for images, labels in train_loader:        
        # Transfer variables to the default device
        images = images.to(device)
        labels = labels.to(device)

        # Clear optimizer of previous values
        optimizer.zero_grad()

        # Forward pass
        output = model(images)

        # Calculate loss    
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()
        losses.update(loss.item() , images.size(0))
        #train_losses.append(loss.item())       

    train_loss = losses.avg

    return train_loss


################################################################################
# Computes and stores the average and current value
################################################################################
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



batch_size      = 64
test_batch_size = 32
num_classes     = 4250
epochs          = 300
learning_rate   = 0.0005
img_dim         = 224

workers         = 3 
save_frequency  = 50

run             = 5
arch            = 101
dropout_flag    = True


train_file      = "splits/train_5.7k_cropped_remapped.txt"
val_file        = "splits/val_5.7k_cropped_remapped.txt"


folder          = "models/run"
save_dir        = os.path.join(folder + "_" + str(run))

if (os.path.isdir(save_dir)):
    print(save_dir + " Folder exists")
else:
    print("Creating " + save_dir)
    os.mkdir(save_dir)

log_file         = os.path.join(save_dir, "train.log")

print("\n===============================================================================\n");
print("Train file         = {}"    .format(train_file))
print("Val file           = {}"    .format(val_file))
print("Run No             = {}"    .format(run))
print("Model save folder  = {}"    .format(save_dir))
print("-------------------------------")
print("Optimisation Parameters")
print("-------------------------------")
print("lr                 = {:.5f}".format(learning_rate))
print("epochs             = {}"    .format(epochs))
print("Batch size         = {}"    .format(batch_size))
print("Test Batch size    = {}"    .format(test_batch_size))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))


########################################################################################
# Get the datasplits in place
########################################################################################
# random brightness, Gaussian noise, random crops, and random blur.
# https://towardsdatascience.com/a-gold-winning-solution-review-of-kaggle-humpback-whale-identification-challenge-53b0e3ba1e84
transformations  = transforms.Compose([transforms.ToTensor()])
#transformations2  = transforms.Compose([transforms.RandomAffine(degrees = 12, translate=(0.036, 0.036), scale=(0.9, 1.1)), transforms.ColorJitter(0.2,0.2,0.2,0.02), transforms.ToTensor()])
#print(transformations2)

dataset_from_csv  = CustomDatasetFromCSVTrain(train_file, height = img_dim, width = img_dim, transforms = transformations)
train_loader      = torch.utils.data.DataLoader(dataset=dataset_from_csv  ,  batch_size= batch_size, shuffle=True, num_workers = workers)

dataset_from_csv2 = CustomDatasetFromCSV      (val_file, height = img_dim, width = img_dim, transforms = transformations)
val_loader        = torch.utils.data.DataLoader(dataset=dataset_from_csv2,  batch_size= test_batch_size, shuffle=False, num_workers = workers)

########################################################################################
# Get the model
########################################################################################
if (arch == 101):
    print("Using resnet 101")
    model = models.resnet101(pretrained=True)
elif (arch == 50):
    print("Using resnet 50")    
    model = models.resnet50(pretrained=True)
else:
    print("Unknown model")



if dropout_flag:
    print("Using Dropout")
    model.fc = nn.Sequential(Flatten(), nn.Dropout(0.25), nn.Linear(2048, num_classes))
else:
    model.fc = nn.Sequential(Flatten(), nn.Linear(2048, num_classes))

model.to(device)
#print("\nDetailed report of new model parameters")
#summary(model,(3,224,224))

########################################################################################
# Get the loss function
########################################################################################
criterion = nn.CrossEntropyLoss()

########################################################################################
# Get the optimizer
########################################################################################
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs):
    train_loss  = train_epoch(model, criterion, optimizer, device, train_loader)
    score       = get_score_model(model, val_loader)
        
    # Printing after each epoch
    print("Epoch: {} Training_Loss: {:.6f} Validation_Acc: {:.3f}".format(epoch + 1, train_loss, score))

    if ( (epoch+1)% save_frequency == 0):
        torch.save(model, os.path.join(save_dir, "checkpoint_" + str(epoch+1) + ".pt"))

