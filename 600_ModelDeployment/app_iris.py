#%% package
from flask import Flask, request
from model_class import MultiClassNet
import torch
import json
#%% model instance
model = MultiClassNet(HIDDEN_FEATURES=6, NUM_CLASSES=3, NUM_FEATURES=4)
local_file_path = '600_ModelDeployment/model_iris.pt'
model.load_state_dict(torch.load(local_file_path))

#%%
app = Flask(__name__)

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return 'Please use POST method'
    if request.method == 'POST':
        # data = request.data.decode('utf-8')
        # print(data)
        # dict_data = json.loads(data.replace("'", "\""))
        X = torch.tensor([[5.1, 4.3, 2.3, 1.8]])
        y_test_hat_softmax = model(X)
        y_test_hat = torch.max(y_test_hat_softmax, 1)
        y_test_cls = y_test_hat.indices.cpu().detach().numpy()[0]
        cls_dict = {
            0: 'setosa', 
            1: 'versicolor', 
            2: 'virginica'
        }
        return f"Your flower belongs to class {cls_dict[y_test_cls]}"


if __name__ == '__main__':
    app.run()
# %% not working but moving on
