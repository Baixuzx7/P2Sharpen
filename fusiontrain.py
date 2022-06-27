from model import TransferNetwork
from model import FusionNetwork
from dataset import PanSharpeningDataset
import torch.nn as nn
import torchvision
import argparse
import torch.utils.data
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from Eval import *


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--decay', type=float, default=0.9)
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument('--stddev', type=float, default=1e-3)
parser.add_argument('--down_sample_size',type = int,default=128)
opt = parser.parse_args()
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
print('Device :', device)
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
train_dataset = PanSharpeningDataset('./Dataset/TrainFolder', transform=data_transform['train'])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
print(len(train_dataset))
valid_dataset = PanSharpeningDataset('./Dataset/ValidFolder', transform=data_transform['test'])
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset))
print(len(valid_dataset))
net = FusionNetwork()
if os.path.exists('./Model/P2Net/fusion.pth'):
    print('Fusion net is loading parameters')
    net.load_state_dict(torch.load('./Model/P2Net/fusion.pth',map_location=device))
else:
    print('Fusion net is initializing')
    net.initialize()
    pass
net.to(device)
transfer = TransferNetwork()
if os.path.exists('./Model/STNet/transfer.pth'):
    print('Transfer net is loading parameters')
    transfer.load_state_dict(torch.load('./Model/STNet/transfer.pth',map_location=device))
else:
    print('Transfer net loading Error')
    exit(0)
transfer.to(device)
loss_function = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
writer = SummaryWriter("./runs/fusion_eval")
QI = QualityIndex(sensor='QB')
# Train
for epoch in range(opt.n_epochs):
    # train
    net.train()
    transfer.eval()
    print('Training')
    for i, (image_pan, image_ms, image_pan_label, image_ms_label) in enumerate(train_loader):
        image_pan = image_pan.to(device)
        image_ms = image_ms.to(device)
        image_pan_label = image_pan_label.to(device)
        image_ms_label = image_ms_label.to(device)

        # Reduced-Resolution Fusion Stage
        optimizer.zero_grad()
        ms1 = net(image_pan,image_ms)
        ms1_ds =  F.interpolate(ms1,size=(32,32), mode='bicubic', align_corners=False) 
        pan1 = transfer(ms1)
        loss_hr = loss_function(ms1,image_ms_label)
        loss_lr = loss_function(ms1_ds,image_ms)
        loss_rst = loss_function(pan1,image_pan)
        loss1 = loss_hr + 0.5 * loss_lr  + 0.5 * loss_rst
        loss1.backward()
        optimizer.step()

        # Full-Resolution Fusion Stage
        optimizer.zero_grad()
        ms2 = net(image_pan_label,ms1.clone().detach())
        ms2_ds = F.interpolate(ms2,size=(128,128), mode='bicubic', align_corners=False)
        pan2 = transfer(ms2)
        loss_rr = loss_function(ms2_ds,image_ms_label)
        loss_hst = loss_function(pan2,image_pan_label)
        loss2 = loss_rr + 0.5 * loss_hst
        loss2.backward()
        optimizer.step()

        print('Epoch : {}/{}   Batch : {}/{}  Loss1 : [{}]  Loss2 : [{}]'.format(epoch, opt.n_epochs, i, len(train_loader)
                                                                        , loss1.item(), loss2.item()))
    # Save
    torch.save(net.state_dict(), './Model/P2Net/{}.pth'.format(epoch))
    # Evaluation
    print('Evaluating')
    net.eval()

    # Reduced-Resolution
    er,rm,ra,qa = [],[],[],[]
    for j in range(len(valid_dataset)):
        image_pan, image_ms, image_pan_label, image_ms_label = valid_dataset[j]
        with torch.no_grad():
            lr_tf = net(image_pan.unsqueeze(0).to(device),image_ms.unsqueeze(0).to(device))
            lr = TensorToIMage(lr_tf.squeeze(0).cpu())
            ms = TensorToIMage(image_ms_label)
            er.append(ERGAS(ms,lr))
            rm.append(RMSE(ms,lr))
            ra.append(RASE(ms,lr))
            qa.append(QAVE(ms,lr))
            pass
        pass
    er_mean = np.asarray(er).mean()
    rm_mean = np.asarray(rm).mean()
    ra_mean = np.asarray(ra).mean()
    qa_mean = np.asarray(qa).mean()

    # Full-Resolution
    dl,ds,qnr = [],[],[]
    for j in range(len(valid_dataset)):
        image_pan, image_ms, image_pan_label, image_ms_label = valid_dataset[j]
        with torch.no_grad():
            hr_tf = net(image_pan_label.unsqueeze(0).to(device),image_ms_label.unsqueeze(0).to(device))
            hr = TensorToIMage(hr_tf.squeeze(0).cpu())
            ms = TensorToIMage(image_ms_label)
            pan = TensorToIMage(image_pan_label)
            D_lambda_index,D_s_index,QNR_index = QI.QNR(hr,ms,pan)
            dl.append(D_lambda_index)
            ds.append(D_s_index)
            qnr.append(QNR_index)
            pass
        pass
    dl_mean = np.asarray(dl).mean()
    ds_mean = np.asarray(ds).mean()
    qnr_mean = np.asarray(qnr).mean()
    # Summary and Write
    writer.add_scalar('ERGAS',er_mean,global_step=epoch)
    writer.add_scalar('RMSE',rm_mean,global_step=epoch)
    writer.add_scalar('RASE',ra_mean,global_step=epoch)   
    writer.add_scalar('QAVE',qa_mean,global_step=epoch)
    writer.add_scalar('QNR ',dl_mean,global_step=epoch)
    writer.add_scalar('Ds  ',ds_mean,global_step=epoch)
    writer.add_scalar('Dl  ',qnr_mean,global_step=epoch)
print('Finished')
