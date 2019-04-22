#!/usr/bin/python3

"""
    ./visualise_transformations.py -i /ec_whale/splits/val_remapped.txt 

    Version 1 Abhinav Kumar 2019-04-16
"""

import argparse
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './src')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torchvision              import transforms
from CustomDatasetFromCSV     import CustomDatasetFromCSV


################################################################################
# Argument Parsing
################################################################################
ap = argparse.ArgumentParser()
ap.add_argument    ('-i', '--input', help = 'path of the input file', default='/ec_whale/splits/val_remapped.txt')
args = ap.parse_args()


################################################################################
# Main function
################################################################################
def main():
    workers         = 3
    img_dim         = 224
    test_repeat     = 6     # No of times augmentation to be done
    test_batch_size = 512   
    rows            = 5     # No of rows/samples to visualize

    sep_for_csv     = ","

    input_file      = args.input
        
    transformations  = transforms.Compose([transforms.ToTensor()])
    dataset_from_csv = CustomDatasetFromCSV(input_file, img_dim, img_dim, transformations)
    val_loader       = torch.utils.data.DataLoader(dataset=dataset_from_csv,  batch_size=test_batch_size, shuffle=False)

    original    = np.zeros((len(dataset_from_csv), 3, img_dim, img_dim))
    transformed = np.zeros((test_repeat, len(dataset_from_csv), 3, img_dim, img_dim))
    print()

    for i, (data,target) in enumerate(val_loader):
        if (i == 0):
            temp = data.detach().cpu()
        else:
            temp = torch.cat((temp, data.detach().cpu()), 0)
    
    original = np.asarray(temp)

    #transformations2  = transforms.Compose([transforms.ColorJitter( brightness=0.2, contrast=0.5, saturation=0.2), transforms.ToTensor()])
    transformations2  = transforms.Compose([transforms.RandomAffine(degrees = 12, translate=(0.036, 0.036), scale=(0.9, 1.1)), transforms.ColorJitter(0.2,0.2,0.2,0.02), transforms.ToTensor()])
    dataset_from_csv2 = CustomDatasetFromCSV(input_file, img_dim, img_dim, transformations2)
    val_loader2       = torch.utils.data.DataLoader(dataset=dataset_from_csv2,  batch_size=test_batch_size, shuffle=False)

    print("Doing test augmentation {} times".format(test_repeat))
    for j in range(test_repeat):
        print("Pass {} done".format(j+1))            
        for i, (data,target) in enumerate(val_loader2):
            if (i == 0):
                temp = data.detach().cpu()
            else:
                temp = torch.cat((temp, data.detach().cpu()), 0)
            
        transformed[j] = np.asarray(temp)

    np.random.seed(seed=1)
    indices = np.random.randint(len(dataset_from_csv), size = rows)

    # Visualise the samples
    columns = test_repeat + 1
    fig = plt.figure()
    # ax enables access to manipulate each of subplots
    ax = []
    for i in range(rows):
        for j in range(columns):
            # create subplot and append to ax
            ax.append( fig.add_subplot(rows, columns, i*columns+ j + 1) )
            
            if (j == 0):
                temp = original[indices[i]]
                title = "Original"
            else:
                temp = transformed[j-1,indices[i]]
                title = "Augmented_" + str(j)

            ax[-1].set_title(title)
            temp = temp*255
            temp = temp.astype(int)
            temp = temp.transpose(1, 2, 0)
            plt.imshow(temp)
            plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
