#!/usr/bin/python3

import os
import argparse
import glob
import numpy as np
import random
from datetime import datetime

################################################################################
# Argument Parsing
################################################################################
description = "Creates a label map which brings the labels from zero indexed."
ap = argparse.ArgumentParser(description = description)
ap.add_argument    ('-i', '--input', help = 'path of the input file', default='/ec_whale/splits/train_val.csv')
args = ap.parse_args()

input_file = args.input
sep_output = " "
sep_input  = ","
output_file = input_file[0:-4] + "_remapped.txt"


prefix = "/ec_whale/data/train"

label_list = []
cnt = 0
with open(input_file) as f:
    for line in f:
        cnt += 1
        if (cnt > 1):        
            line = line.strip()
            name, label = line.split(sep_input)

            if (label in label_list or label == "new_whale"):
                pass
            else:
                label_list.append(label)

#print(label_list)

num_uniq_whales = len(label_list)
print("Number of unique whales = {}".format(num_uniq_whales))
new_label_arr   = np.arange(num_uniq_whales)

# Write label map first
label_map_path = os.path.join(os.path.dirname(input_file), "label_map.txt")
print("\nLabel Map Path = {}".format(label_map_path))
print("Writing label map now...")
"""
with open(label_map_path, "w") as fw:
    fw.write("Label_Original" + sep_output + "Label_New")    
    for i in range(num_uniq_whales):
        fw.write(label_list[i] + sep_output + str(new_label_arr[i]) + "\n")

print("Done")
"""

print(output_file)

# Read the input csv file
with open(input_file) as f: 
    li = f.readlines()

print(li[0], li[1])

# delete header
del li[0]

print(li[0], li[1])
# shuffle the list
random.shuffle(li)

"""
with open(output_file, "w") as fw:
    for i in range(len(li)):
        line        = li[i].strip()
        name, label = line.split(sep_input)
        if (label in label_list):           
            # search which index the label list is
            index       = label_list.index(label)
            fw.write(os.path.join(prefix,name) + sep_output + str(index) + "\n")
        
""" 
