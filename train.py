#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(description='Process data for emotion selection.')
parser.add_argument('--data_directory', type=str, default = './TrainTestFiles',required=True,
                    help='Directory containing data and labels')
parser.add_argument('--emotion', type=int, default=3,
                    help='Emotion to select (0=Arousal, 1=Valence, 2=Dominance, 3=Like)')

args = parser.parse_args()

if not os.path.exists(args.data_directory):
    raise FileNotFoundError(f"Data directory '{args.data_directory}' not found. Please provide a valid data directory.")

train_data_file = os.path.join(args.data_directory, 'data_training.npy')
train_label_file = os.path.join(args.data_directory, 'label_training.npy')
test_data_file = os.path.join(args.data_directory, 'data_testing.npy')
test_label_file = os.path.join(args.data_directory, 'label_testing.npy')

# Check if the specified files exist
for file_path in [train_data_file, train_label_file, test_data_file, test_label_file]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please provide a valid file path.")

# Load training data and labels
with open(train_data_file, 'rb') as fileTrain:
    X = np.load(fileTrain)

with open(train_label_file, 'rb') as fileTrainL:
    Y = np.load(fileTrainL)

# Normalize the data
X = normalize(X)

# Select the specified emotion
print(args.emotion)
Z = np.ravel(Y[:, [args.emotion]])
print(Z)

Z= Z-1

x_train = np.array(X[:])
#y_train = to_categorical(Z)
y_train = to_categorical(Z, num_classes=9)

# Load testing data and labels
with open(test_data_file, 'rb') as fileTest:
    M = np.load(fileTest)

with open(test_label_file, 'rb') as fileTestL:
    N = np.load(fileTestL)

M = normalize(M)

# Select the specified emotion
L = np.ravel(N[:, [args.emotion]])

# Prepare testing data and labels
x_test = np.array(M[:])
L = L-1
#y_test = to_categorical(L)
y_test = to_categorical(L, num_classes=9)



scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)

# Determine if CUDA is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using GPU device 0")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Convert NumPy arrays to PyTorch tensors
x_train = torch.Tensor(x_train).to(device)
y_train = torch.Tensor(y_train).to(device)
x_test = torch.Tensor(x_test).to(device)
y_test = torch.Tensor(y_test).to(device)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


class CustomCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.fc1 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


input_shape = (128, 1) 
num_classes = 9 
model = CustomCNN(input_shape, num_classes)
model = model.to(device)

print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_test, y_test)

# Convert datasets to DataLoaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

train_losses, train_accuracies, train_f1_scores = [], [], []
val_losses, val_accuracies, val_f1_scores = [], [], []


# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss, train_correct, train_total = 0.0, 0, 0
    all_train_labels, all_train_predictions = [], []

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", unit="batch")
    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels_indices = torch.max(labels, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels_indices).sum().item()

        all_train_labels.extend(labels_indices.cpu().numpy())
        all_train_predictions.extend(predicted.cpu().numpy())

        train_loader_tqdm.set_postfix(loss=running_loss/len(train_loader_tqdm), accuracy=100.0 * train_correct / train_total)

    train_f1 = f1_score(all_train_labels, all_train_predictions, average='weighted')
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100.0 * train_correct / train_total)
    train_f1_scores.append(train_f1)


    model.eval() 
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_val_labels, all_val_predictions = [], []

    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]", unit="batch")
    with torch.no_grad():
        for inputs, labels in val_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, labels_indices = torch.max(labels, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels_indices).sum().item()

            all_val_labels.extend(labels_indices.cpu().numpy())
            all_val_predictions.extend(predicted.cpu().numpy())

            val_loader_tqdm.set_postfix(loss=val_loss/len(val_loader_tqdm), accuracy=100.0 * val_correct / val_total)

    val_f1 = f1_score(all_val_labels, all_val_predictions, average='weighted')
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(100.0 * val_correct / val_total)
    val_f1_scores.append(val_f1)


final_training_accuracy = train_accuracies[-1]
final_validation_accuracy = val_accuracies[-1]
average_training_accuracy = sum(train_accuracies) / len(train_accuracies)
average_validation_accuracy = sum(val_accuracies) / len(val_accuracies)

# Printing or logging the accuracies
print(f"Final Training Accuracy: {final_training_accuracy}%")
print(f"Final Validation Accuracy: {final_validation_accuracy}%")
print(f"Average Training Accuracy: {average_training_accuracy}%")
print(f"Average Validation Accuracy: {average_validation_accuracy}%")

final_train_f1 = train_f1_scores[-1]
final_val_f1 = val_f1_scores[-1]
average_train_f1 = np.mean(train_f1_scores)
average_val_f1 = np.mean(val_f1_scores)

# Printing or logging the F1 scores
print(f"Final Training F1 Score: {final_train_f1}")
print(f"Final Validation F1 Score: {final_val_f1}")
print(f"Average Training F1 Score: {average_train_f1}")
print(f"Average Validation F1 Score: {average_val_f1}")

torch.save(model, r"./model_Dominance.pt")

