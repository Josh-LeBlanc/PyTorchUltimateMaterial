#%% packages
import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

#%% Data Prep
image_path = 'kiki.jpg'
image = Image.open(image_path)
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

X = transformations(image).unsqueeze(0)
X.shape
# %%
model = resnet18(pretrained=True)

# %%
class MyHook:
    def __init__(self):
        self.layer_out = []
        self.layer_shape = []

    def __call__(self, module, module_in, module_out):
        self.layer_out.append(module_out)
        self.layer_shape.append(module_out.shape)
# %%
hook = MyHook()
for l in model.modules():
    if isinstance(l, torch.nn.modules.conv.Conv2d):
        handle = l.register_forward_hook(hook)
# %%
y_pred = model(X)
# %%
len(hook.layer_out)
# %%
layer_num = 0
layer_imgs = hook.layer_out[layer_num].detach().numpy()
# %%
for i in range(4):
    plt.imshow(layer_imgs[0, i, :, :])
    plt.show()
# %%