#%% Packages
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

# %% Data Import
data = sns.load_dataset("flights")
print(f'Number of Entries: {len(data)}')
data.head()

# %%
sns.lineplot((data.index, data.passengers))
# %%
# Convert passenter data to float32 for PyTorch
num_points = len(data)
Xy = data.passengers.values.astype(np.float32)

#%% scale the data
scaler = MinMaxScaler()

Xy_scaled = scaler.fit_transform(Xy.reshape(-1, 1))


# %% Data Restructuring

#%% train/test split

# TODO: create train and test set (keep last 12 months for testing, everything else for training)
X_restruct = []
y_restruct = []

for i in range(num_points - 10):
    X_restruct.append(Xy_scaled[i:i+10])
    y_restruct.append(Xy_scaled[i+10])

X_restruct = np.array(X_restruct)
y_restruct = np.array(y_restruct)

X_test = X_restruct[len(X_restruct) - 12:]
X_train = X_restruct[:len(X_restruct) - 12]
y_test = y_restruct[len(X_restruct) - 12:]
y_train = y_restruct[:len(X_restruct) - 12]
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# TODO: restructure the data to match the following shapes
# X_train shape: (122, 10, 1)
# y_train shape: (122, 1)
# X_test shape: (12, 10, 1)
# y_test shape: (12, 1)


# %% 
#  TODO: create dataset and dataloader
class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, x):
        return self.X[x], self.y[x]
    
train_loader = DataLoader(FlightDataset(X_train, y_train), batch_size=2)
test_loader = DataLoader(FlightDataset(X_test, y_test), batch_size=len(y_test))

# %%
# TODO: set up the model class
class FlightModel(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(FlightModel, self).__init__()
        self.hidden_size = 50
        self.lstm = nn.LSTM(input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(torch.relu(x))
        return x

# %% Model, Loss and Optimizer
model = FlightModel()

loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
NUM_EPOCHS = 200

#%% Train
# TODO: create the training loop
for epoch in range(NUM_EPOCHS):
    for j, data in enumerate(train_loader):
        X, y = data
        optimizer.zero_grad()
        y_pred = model.forward(X)
        loss = loss_fun(y_pred, y)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.data}")


# %% Create Predictions
test_set = FlightDataset(X_test, y_test)
X_test_torch, y_test_torch = next(iter(test_loader))
with torch.no_grad():
    y_pred = model(X_test_torch)
y_act = y_test_torch.numpy().squeeze()
x_act = range(y_act.shape[0])
sns.lineplot(x=x_act, y=y_act, label = 'Actual',color='black')
sns.lineplot(x=x_act, y=y_pred.squeeze(), label = 'Predicted',color='red')

# %% correlation plot
sns.scatterplot(x=y_act, y=y_pred.squeeze(), label = 'Predicted',color='red', alpha=0.5)

# %%
