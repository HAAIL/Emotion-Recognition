#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import argparse
import os

subjectList = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
               '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
               '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_inputdirec(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f'{directory} not found enter a valid path')

parser = argparse.ArgumentParser(description='Process some EEG data.')

parser.add_argument('--input_directory',type=str,default='./pre_data',help = "input directory for the pre processed data")

parser.add_argument('--num_subjects', type=int, default=len(subjectList),
                    help='Number of subjects to select (default: all subjects)')
parser.add_argument('--output_directory', type=str, default='./Data_selected',
                    help='Output directory for the selected subjects')
#parser.add_argument('--input_directory',type=str,default='./pre_data',help = "input directory for the pre processed data")

args = parser.parse_args()

selected_subjects = subjectList[:args.num_subjects]

check_inputdirec(args.input_directory)
# Create the output directory
create_directory(args.output_directory)

data = []
labels = []

# Load data for each selected subject and append to data and labels lists
for subject in selected_subjects:
    with open(os.path.join(args.input_directory,f's{subject}.npy'),'rb') as file:
    #with open(f'./pre_data/s{subject}.npy', 'rb') as file:
        sub = np.load(file, allow_pickle=True)
        for trial in sub:
            data.append(trial[0])   # EEG data
            labels.append(trial[1]) # Corresponding labels

# Save the selected data and labels as .npy files in the specified directory
np.save(os.path.join(args.output_directory, 'selected_data'), np.array(data), allow_pickle=True, fix_imports=True)
np.save(os.path.join(args.output_directory, 'selected_labels'), np.array(labels), allow_pickle=True, fix_imports=True)

# Print out the shapes of the saved data and labels to confirm
print("Selected dataset:", np.array(data).shape, np.array(labels).shape)

