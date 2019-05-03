#!/usr/bin/python3

# Taken from
# https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb


import os
import argparse
import glob
import numpy as np
import random
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageOps

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils

import torch
from torch.autograd import Variable
import torch.nn.functional as F  
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader,Dataset
from torchsummary import summary

import sys
sys.path.insert(0, './src')
from SiameseNetworkDataset import SiameseNetworkDataset
from CustomDatasetFromCSV  import CustomDatasetFromCSV


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(401408, 500),
            #8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive



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

    for images1, images2, labels in train_loader:        
        # Transfer variables to the default device
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels  = labels.float().to(device)

        # Clear optimizer of previous values
        optimizer.zero_grad()

        # Forward pass
        output1, output2 = model(images1, images2)

        # Calculate constrastive loss    
        loss = criterion(output1, output2, labels)
        loss.backward()

        optimizer.step()
        losses.update(loss.item() , images1.size(0))       

    train_loss = losses.avg

    return train_loss


################################################################################
# Validating function
################################################################################
def val(model, device, train_loader, val_loader):

    # Transfer model to the default device
    model  = model.to(device)

    model.eval()

    len_train = len(train_loader.dataset)
    len_val   = len(val_loader.dataset)

    feat_train  = np.zeros((len_train, feature_dim))
    label_train = np.zeros((len_train,))
    feat_val    = np.zeros((len_val  , feature_dim))
    label_val   = np.zeros((len_val  ,))
    
    end = 0
    for images1, labels in train_loader:
        images2 = torch.zeros(images1.shape)
        
        # Transfer variables to device
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels  = labels.to(device)

        output1, _ = model(images1, images2)

        start = end
        end  += images1.shape[0]

        feat_train [start:end, :] = output1.data.cpu().numpy()
        label_train[start:end]    =  labels.data.cpu().numpy()

    print("Train images done")
    
    end=0
    for images1, labels in val_loader:
        images2 = torch.zeros(images1.shape)
        
        # Transfer variables to device
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels  = labels.to(device)

        output1, _ = model(images1, images2)

        start = end
        end  += images1.shape[0]
        feat_val [start:end, :] = output1.data.cpu().numpy()
        label_val[start:end]    =  labels.data.cpu().numpy()

    print("Val images done")

    cnt = 0
    correct = 0

    # start matching
    for i in range(len_val):
        if (label_val[i] in label_train):
            cnt += 1

            # Get distances
            temp = feat_train - feat_val[0]
            dist = np.linalg.norm(temp,axis=1)
            
            min_index  = np.argmin(dist)
            predicted  = label_train[min_index]
            ground     = label_val  [i]

            if (ground == predicted):
                correct += 1
        else:
            pass

        if (i % 100 == 0):
            print("{} Images with labels processed. {} correct. Accuracy = {:.2f}".format(cnt, correct, 100.*correct/cnt))


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



########################################################################################
# Parameters
########################################################################################
batch_size      = 64
test_batch_size = 32
num_classes     = 4250
epochs          = 100
learning_rate   = 0.0005
img_dim         = 224

workers         = 3 
save_frequency  = 2

run             = 11
arch            = 50
dropout_flag    = True

feature_dim     = 5

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

dataset_from_csv  = CustomDatasetFromCSV(train_file, height = img_dim, width = img_dim, transforms = transformations)
train_loader      = torch.utils.data.DataLoader(dataset=dataset_from_csv  ,  batch_size= batch_size, shuffle=False, num_workers = workers)

dataset_from_csv2 = CustomDatasetFromCSV      (val_file, height = img_dim, width = img_dim, transforms = transformations)
val_loader        = torch.utils.data.DataLoader(dataset=dataset_from_csv2,  batch_size= test_batch_size, shuffle=False, num_workers = workers)

########################################################################################
# Get the model
########################################################################################
model = SiameseNetwork()

model.load_state_dict(torch.load( os.path.join(save_dir, "checkpoint_66.pt")))

model = model.cuda()
#"""
summary(model,[(3,224,224),(3,224,224)])

val(model, device, train_loader, val_loader)
