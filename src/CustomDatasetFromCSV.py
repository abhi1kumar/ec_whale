"""
    Making a custom dataloader in Pytorch

    Version 1 Abhinav Kumar 2018-03-26

    Taken from
    https://github.com/utkuozbulak/pytorch-custom-dataset-examples
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import imgaug.augmenters as iaa

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height=224, width=224, transforms=[transforms.ToTensor()]):
        """
        Args:
            csv_path (string): path to csv file
            height (int)     : image height
            width (int)      : image width
            transform        : pytorch transforms for transforms and tensor conversion
        """

        # Read the csv file separated by whitespace
        self.data_info = pd.read_csv(csv_path, delim_whitespace=True, header=None)

        # First column has paths, second column has labels
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        self.data_len  = len(self.data_info.index)

        # Optional Arguments
        self.height = height
        self.width  = width
        self.transforms = transforms

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]

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

        # Transform image to tensor by passsing through the composed version of transforms
        if self.transforms is not None:
            img_as_tensor  = self.transforms(img_padded)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


    def aug_image(self, image, is_infer=False, augment = None):
        if is_infer:
            flip_code = augment[0]

            if flip_code == 1:
                seq = iaa.Sequential([iaa.Fliplr(1.0)])
            elif flip_code == 2:
                seq = iaa.Sequential([iaa.Flipud(1.0)])
            elif flip_code == 3:
                seq = iaa.Sequential([iaa.Flipud(1.0),
                                      iaa.Fliplr(1.0)])
            elif flip_code ==0:
                return image

        else:

            seq = iaa.Sequential([
                iaa.Affine(rotate= (-15, 15),
                           shear = (-15, 15),
                           mode='edge'),

                iaa.SomeOf((0, 2),
                           [
                               iaa.GaussianBlur((0, 1.5)),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5),
                               iaa.AddToHueAndSaturation((-5, 5)),  # change hue and saturation
                               iaa.PiecewiseAffine(scale=(0.01, 0.03)),
                               iaa.PerspectiveTransform(scale=(0.01, 0.1))
                           ],
                           random_order=True
                           )
            ])

        image = seq.augment_image(image)
        return image

