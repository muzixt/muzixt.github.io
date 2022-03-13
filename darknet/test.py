import os
import argparse
import time
import shutil

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
import torchvision.datasets as datasets
import numpy as np
from model import darknet53
from net2 import *
from train import evaluate, predict

#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
ds_test = datasets.ImageFolder("./data/test/",
                               transform=transform_test, target_transform=lambda t: torch.tensor([t]).float())

testLoader = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=8)

class_to_idx = ds_test.class_to_idx

class_to_nums = {k: ds_test.targets.count(v) for k, v in class_to_idx.items()}

print(class_to_idx)
print(class_to_nums)

model = resnet34(num_classes=5).to(device)

state = torch.load("./log/model/Best_2022-03-12-19-57-00-116956_model.pth")
model.load_state_dict(state['state_dict'], strict=False)

pre, target = predict(model, testLoader)
_, pre = pre.topk(1, 1)
target = target.squeeze(1).cpu().detach().numpy().astype(int)
pre = pre.squeeze(1).cpu().detach().numpy().astype(int)

q = 0
counts = 0
for (k, v), (k1, v1) in zip(class_to_nums.items(), class_to_idx.items()):
    q += v
    count = np.sum(pre[0:q] == v1)
    counts += count
    print(f"{k} acc :{count / v:.3f}")
else:

    print(f"total acc :{counts / q:.3f}")
