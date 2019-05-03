#!/usr/bin/python3

import os
import argparse
import glob
import numpy as np
import random
import pandas as pd

sep_csv = " "
input_file = "splits/train_5.7k_cropped_remapped.txt"
positive_flag = 0
negative_flag = 1 - positive_flag


df = pd.read_csv(input_file, sep=sep_csv, header = None)
labels = (df.iloc[:, 1].values)
print(labels)

with open(input_file[:-4] + "_siamese.txt", "w") as f:
    
    # for each element of the dataset
    for i in range(df.shape[0]):
        label = labels[i]
        
        # get two positives first
        ind, = np.where(labels == label)

        #print(ind)
        # remove the element itself
        ind  = np.delete(ind, np.where(ind == i))
        #print(ind)
        np.random.shuffle(ind)

        #print(ind)
        # Throw the sample since contains no positives
        if (ind.shape[0] < 1):
            #print("Only single image of this class")
            pass
        else:                
            for j in range(np.min([2, ind.shape[0]])):
                f.write(str(df.iloc[i,0]) + sep_csv + str(df.iloc[ind[j],0]) + sep_csv + str(positive_flag) + "\n")
    
            # get two negatives then
            ind,  = np.where(labels != label)
            np.random.shuffle(ind)

            # Throw the sample since contains no positives      
            for j in range(np.min([2, ind.shape[0]])):
                f.write(str(df.iloc[i,0]) + sep_csv + str(df.iloc[ind[j],0]) + sep_csv + str(negative_flag) + "\n")  

        if ((i+1) % 1000 == 0):
            print("{} Images done.".format(i+1))
