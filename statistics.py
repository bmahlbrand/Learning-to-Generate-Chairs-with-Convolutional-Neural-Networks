"""
This script is for getting the mean and std of three channels of the datasets. 
"""

import numpy as np
from skimage import io
import torch
import pandas as pd

# get the .csv paths
train_csv_path = "../data/chairs.train.class.csv"
valid_csv_path = "../data/chairs.valid.class.csv"

# read the dataset and append three channels respectively
train_df = pd.read_csv(train_csv_path, header=None)
train_image = np.asarray(train_df.iloc[:, 0])[1:]
valid_df = pd.read_csv(valid_csv_path, header=None)
valid_image = np.asarray(valid_df.iloc[:, 0])[1:]

red = np.ones(1)
green = np.ones(1)
blue = np.ones(1)

for image in train_image:
    image_read = io.imread(image)
    red = np.append(red, image_read[:, :, 0])
    green = np.append(green, image_read[:,:,1])
    blue = np.append(blue, image_read[:,:,2])

for image in valid_image:
    image_read = io.imread(image)
    red = np.append(red, image_read[:, :, 0])
    green = np.append(green, image_read[:,:,1])
    blue = np.append(blue, image_read[:,:,2])  

red = red[1:]
green = green[1:]
blue = blue[1:]

print("red mean: ", np.mean(red))
print("green mean: ", np.mean(green))
print("blue mean: ", np.mean(blue))

print("red std: ", np.std(red))
print("green std: ", np.std(green))
print("blue mean: ", np.std(blue))
