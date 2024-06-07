#%% packages
!pip install seaborn
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns
import numpy as np
from collections import Counter
# %% data prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size = 0.2)


# %% dataset and dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TODO: create instance of dataset
dataset_train = MultilabelDataset(X_train, y_train)
dataset_test = MultilabelDataset(X_test, y_test)

# TODO: create train loader
train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=True)


# %% model
# TODO: set up model class
# topology: fc1, relu, fc2
# final activation function??
class MultiLabelNet(nn.Module):
    def __init__(self, INPUT_LAYER, CLASSES, HIDDEN_LAYER):
        super(MultiLabelNet, self).__init__()
        self.lin1 = nn.Linear(INPUT_LAYER, HIDDEN_LAYER)
        self.lin2 = nn.Linear(HIDDEN_LAYER, CLASSES)
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lin1(x)
        x = self.ReLU(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        return x


# TODO: define input and output dim
input_dim = dataset_train.X.shape[1]
output_dim = dataset_train.y.shape[1]

# TODO: create a model instance
model = MultiLabelNet(input_dim, output_dim, 20)


# %% loss function, optimizer, training loop
# TODO: set up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.01)
losses = []
slope, bias = [], []
number_epochs = 100

# TODO: implement training loop
for epoch in range(number_epochs):
    for j, data in enumerate(train_loader):
        
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_pred = model(data[0])

        # compute loss
        loss = loss_fn(y_pred, data[1])
        
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

    losses.append(loss.item())
    # TODO: print epoch and loss at end of every 10th epoch
    print(f"epoch: {epoch}, loss: {loss.item()}")
    
    
# %% losses
# TODO: plot losses
sns.lineplot(x=range(number_epochs), y=losses)

# %% test the model
# TODO: predict on test set
with torch.no_grad():
    y_test_pred = model(X_test).round()

#%% Naive classifier accuracy
# TODO: convert y_test tensor [1, 1, 0] to list of strings '[1. 1. 0.]'
y_test_str = [str(i) for i in y_test.detach().numpy()]
Counter(y_test_str)

# TODO: get most common class count
most_common_count = Counter(y_test_str).most_common()[0][1]
print(f"Naive Classifier accuracy: {most_common_count / len(y_test_str) * 100}%")

# TODO: print naive classifier accuracy


# %% Test accuracy
# TODO: get test set accuracy
accuracy_score(y_test, y_test_pred)

# %%
