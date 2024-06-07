#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
# %% data import
iris = load_iris()
X = iris.data
y = iris.target
print(X.shape, y.shape)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# %% convert to float32

# %% dataset
class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        # self.y = self.y.type(torch.LongTensor)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

# %% dataloader
iris_data = IrisDataset(X_train, y_train)
train_loader = DataLoader(iris_data, batch_size=32, shuffle=True)

# %% check dims
print(f"X shape: {iris_data.X.shape}, y shape: {iris_data.y.shape}")

# %% define class
class MultiClassNet(nn.Module):
    # lin1, lin2, softmax
    def __init__(self, FEATURES, CLASSES, HIDDEN_FEATURES):
        super(MultiClassNet, self).__init__()
        self.lin1 = nn.Linear(FEATURES, HIDDEN_FEATURES)
        self.lin2 = nn.Linear(HIDDEN_FEATURES, CLASSES)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = self.softmax(x)
        return x

# %% hyper parameters
NUM_FEATURES = iris_data.X.shape[1]
HIDDEN = 6
NUM_CLASSES = len(iris_data.y.unique())
# %% create model instance
model = MultiClassNet(NUM_FEATURES, NUM_CLASSES, HIDDEN)


# %% loss function
criterion = nn.CrossEntropyLoss()
# %% optimizer
LR = .01
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# %% training
losses = []
NUM_EPOCHS = 10000

for epoch in range(NUM_EPOCHS):
    for X, y in train_loader:
        # clear gradients
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)

        # calculate loss
        loss = criterion(y_pred, y)

        # calculate gradients
        loss.backward()

        # backpropagate
        optimizer.step()
    losses.append(float(loss.data.detach().numpy()))
     
# %% show losses over epochs
sns.lineplot(x=range(NUM_EPOCHS), y=losses)


# %% test the model
X_test_torch = torch.from_numpy(X_test)
with torch.no_grad():
    y_test_pred = model(X_test_torch)
    y_test_pred = torch.max(y_test_pred.data, 1)


# %% Accuracy
accuracy_score(y_test, y_test_pred.indices)

# %%
from collections import Counter
Counter(y_test)
# %%
