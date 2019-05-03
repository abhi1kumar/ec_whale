import numpy as np
import pandas as pd

from PIL import Image, ImageOps

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import imgaug.augmenters as iaa

class SiameseNetworkDataset(Dataset):
    
    def __init__(self, csv_path, height=224, width=224, transforms=[transforms.ToTensor()]):

        # Read the csv file separated by whitespace
        self.data_info = pd.read_csv(csv_path, delim_whitespace=True, header=None)

        # First column has paths, second column has paths and third has the labels
        self.image_arr   = np.asarray(self.data_info.iloc[:, 0])
        self.image_arr_2 = np.asarray(self.data_info.iloc[:, 1])
        self.label_arr   = np.asarray(self.data_info.iloc[:, 2])
        self.data_len    = len(self.data_info.index)
    

        # Optional Arguments
        self.height = height
        self.width  = width
        self.transforms = transforms 
    

    def get_single_image(self, path):
        # Get image name from the pandas df
        single_image_name = path

        # Open image
        img_as_img = Image.open(single_image_name)
        
        # Resize the image
        # https://stackoverflow.com/a/273962
        img_as_img.thumbnail((self.height, self.width), Image.ANTIALIAS)
        img_as_img = img_as_img.convert('RGB')

        # Image can be of different size. Pad zeros around the image
        # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/#resize-and-pad-with-imageops-module
        delta_w = self.width  - img_as_img.size[0]
        delta_h = self.height - img_as_img.size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        img_padded = ImageOps.expand(img_as_img, padding)

        np_im = np.array(img_padded)
        #print(np.max(np_im))
        #np_img_aug = self.aug_image(np_im)
        np_img_aug  = np_im        
        img_padded = Image.fromarray(np_img_aug.astype('uint8'), 'RGB')

        # Transform image to tensor by passsing through the composed version of transforms
        if self.transforms is not None:
            img_as_tensor  = self.transforms(img_padded)

        # Return image and the label
        return img_as_tensor

    
    def __getitem__(self,index):
        path1 = self.image_arr[index]
        path2 = self.image_arr_2[index]
        
        img0 = self.get_single_image(path1)
        img1 = self.get_single_image(path2)

        # Get label(class) of the image based on the cropped pandas column
        label = self.label_arr[index]

        return (img0, img1 , label)

    
    def __len__(self):
        return self.data_len
