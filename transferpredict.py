import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import argparse
import os
from model import FusionNetwork
from dataset import PanSharpeningDataset
from Eval import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


data_transform = {
    'train': torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ]
    ),
    'test': torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ]
    )
}
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
test_dataset = PanSharpeningDataset('./Dataset/TestFolder', transform=data_transform['test'])
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset))
print(len(test_dataset))
net = FusionNetwork()
if os.path.exists('./Model/P2Net/fusion.pth'):
    print('Fusion Net is loading parameters')
    net.load_state_dict(torch.load('./Model/P2Net/fusion.pth',map_location = device))
else:
    print('Fusion Net can not load any .pth file')
    exit(0)
if not os.path.exists('./IMageFolder/FusionFolderLR'):
    os.makedirs('./IMageFolder/FusionFolderLR')
    pass    
if not os.path.exists('./IMageFolder/FusionFolderLRRS'):
    os.makedirs('./IMageFolder/FusionFolderLRRS')
    pass   
if not os.path.exists('./IMageFolder/FusionFolderHR'):
    os.makedirs('./IMageFolder/FusionFolderHR')
    pass
if not os.path.exists('./IMageFolder/FusionFolderHRRS'):
    os.makedirs('./IMageFolder/FusionFolderHRRS')
    pass   
net.to(device)
for j in range(len(test_dataset)):
    # Training
    print(j)
    net.eval()
    image_pan, image_ms, image_pan_label, image_ms_label = test_dataset[j]
    with torch.no_grad():
        image_fusion = net(image_pan.unsqueeze(0).to(device), image_ms.unsqueeze(0).to(device))   
    image_fusion_np = TensorToIMage(image_fusion.cpu().squeeze(0))
    imageio.imwrite('./IMageFolder/FusionFolderLR/{}.tif'.format(j),image_fusion_np)
    imageio.imwrite('./IMageFolder/FusionFolderLRRS/{}.tif'.format(j),RSGenerate(image_fusion_np[:, :, [2, 1, 0]], 1, 1))
    pass

    with torch.no_grad():
        image_fusion = net(image_pan_label.unsqueeze(0).to(device), image_ms_label.unsqueeze(0).to(device))   
    image_fusion_np = TensorToIMage(image_fusion.cpu().squeeze(0))
    imageio.imwrite('./IMageFolder/FusionFolderHR/{}.tif'.format(j),image_fusion_np)
    imageio.imwrite('./IMageFolder/FusionFolderHRRS/{}.tif'.format(j),RSGenerate(image_fusion_np[:, :, [2, 1, 0]], 1, 1))
    pass
print('Finished')










