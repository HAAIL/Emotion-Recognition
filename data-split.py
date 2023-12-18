#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

parser = argparse.ArgumentParser(description='Split data into training and testing sets.')
parser.add_argument('--input_directory', type=str, default='./Data_selected',
                    help='Input directory containing complete data and labels')
parser.add_argument('--output_directory', type=str, default='./TrainTestFiles',
                    help='Output directory for training and testing sets')

args = parser.parse_args()

# Check if the input directory exists
if not os.path.exists(args.input_directory):
    raise FileNotFoundError(f"Input directory '{args.input_directory}' not found. Please provide a valid input directory.")


create_directory(args.output_directory)


data = np.load(os.path.join(args.input_directory, 'selected_data.npy'), allow_pickle=True)
labels = np.load(os.path.join(args.input_directory, 'selected_labels.npy'), allow_pickle=True)

# Split the data into training and testing sets (you can change the split if you want to)
data_training, data_testing, label_training, label_testing = train_test_split(
    data, labels, test_size=0.15, random_state=42
)

# Save the training and testing sets as .npy files in the specified output directory
np.save(os.path.join(args.output_directory, 'data_training.npy'), data_training, allow_pickle=True, fix_imports=True)
np.save(os.path.join(args.output_directory, 'label_training.npy'), label_training, allow_pickle=True, fix_imports=True)
np.save(os.path.join(args.output_directory, 'data_testing.npy'), data_testing, allow_pickle=True, fix_imports=True)
np.save(os.path.join(args.output_directory, 'label_testing.npy'), label_testing, allow_pickle=True, fix_imports=True)

print("Training dataset:", data_training.shape, label_training.shape)
print("Testing dataset:", data_testing.shape, label_testing.shape)

