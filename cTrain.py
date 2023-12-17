#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import csv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import argparse
import os

parser = argparse.ArgumentParser(description='Process data for emotion selection.')
parser.add_argument('--data_directory', type=str, default = './DataTrainTestFiles',required=True,
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
test_dataset = TensorDataset(x_test, y_test)

# Convert datasets to DataLoaders
#train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

########################################################################################################
n_folds = 2
kf = KFold(n_splits=n_folds, shuffle=True)

# Prepare a CSV file to store the results
custom_directory = './results'
file_name = 'fold_results.csv'

file_path = os.path.join(custom_directory, file_name)

# Create the directory if it doesn't exist
if not os.path.exists(custom_directory):
    os.makedirs(custom_directory)

csv_file = open(f'./results/fold_results{args.emotion}.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Fold', 'Final Train Accuracy', 'Final Val Accuracy', 'Average Train Accuracy', 'Average Val Accuracy', 'Final Train F1', 'Final Val F1', 'Average Train F1', 'Average Val F1'])

# Iterate over each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    print(f"FOLD {fold}")

    # Creating subsets and DataLoaders for the current fold
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    train_loader_fold = DataLoader(train_subset, batch_size=256, shuffle=True)
    val_loader_fold = DataLoader(val_subset, batch_size=256, shuffle=False)
    
    

    # Reset the model for the new fold
    #model = Model()  # Replace with your model
    model = CustomCNN(input_shape, num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.YourOptimizer(model.parameters())  # Replace with your optimizer
    #criterion = YourLossFunction()  # Replace with your loss function

    # Lists for storing metrics for the current fold
    train_losses, train_accuracies, train_f1_scores = [], [], []
    val_losses, val_accuracies, val_f1_scores = [], [], []

    # Training and validation loop
    # (Copy your training and validation code here, make sure to use train_loader_fold and val_loader_fold)
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss, train_correct, train_total = 0.0, 0, 0
        all_train_labels, all_train_predictions = [], []

        train_loader_tqdm = tqdm(train_loader_fold, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", unit="batch")
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
        train_losses.append(running_loss / len(train_loader_fold))
        train_accuracies.append(100.0 * train_correct / train_total)
        train_f1_scores.append(train_f1)


        model.eval() 
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_val_labels, all_val_predictions = [], []

        val_loader_tqdm = tqdm(val_loader_fold, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]", unit="batch")
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
        val_losses.append(val_loss / len(val_loader_fold))
        val_accuracies.append(100.0 * val_correct / val_total)
        val_f1_scores.append(val_f1)
        
        



     
    epochs = range(1, len(train_accuracies) + 1)

    # Plotting training and validation accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

# Plotting training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./results/loss{args.emotion}.png')
    # Computing final and average metrics for the current fold
    final_train_accuracy = train_accuracies[-1]
    final_val_accuracy = val_accuracies[-1]
    average_train_accuracy = sum(train_accuracies) / len(train_accuracies)
    average_val_accuracy = sum(val_accuracies) / len(val_accuracies)
    final_train_f1 = train_f1_scores[-1]
    final_val_f1 = val_f1_scores[-1]
    average_train_f1 = np.mean(train_f1_scores)
    average_val_f1 = np.mean(val_f1_scores)

    # Writing the metrics to the CSV file
    csv_writer.writerow([fold, final_train_accuracy, final_val_accuracy, average_train_accuracy, average_val_accuracy, final_train_f1, final_val_f1, average_train_f1, average_val_f1])

# Close the CSV file
csv_file.close()
torch.save(model, f"./results/model_{args.emotion}.pt")

model.eval()  # Set the model to evaluation mode
all_preds = []
all_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)

        # Convert predictions from one-hot encoded to class indices
        predicted = np.argmax(outputs.cpu().numpy(), axis=1)
        all_preds.extend(predicted)

        # Convert true labels from one-hot encoded to class indices
        true_labels = np.argmax(labels.cpu().numpy(), axis=1)
        all_true.extend(true_labels)


accuracy = accuracy_score(all_true, all_preds)

#print(len(Vval_loader)/256)

print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

cm = confusion_matrix(all_true, all_preds)

# Plotting confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(f'./results/confusion_matrix{args.emotion}.png')



















