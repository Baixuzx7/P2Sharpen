from model import TransferNetwork
from dataset import PanSharpeningDataset
import torch.nn as nn
import torchvision
import argparse
import torch.utils.data
import os
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--decay', type=float, default=0.9)
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument('--stddev', type=float, default=1e-3)
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
net = TransferNetwork()
if os.path.exists('./Model/STNet/transfer.pth'):
    print('Net is loading parameters')
    net.load_state_dict(torch.load('./Model/STNet/transfer.pth'))
else:
    print('Net is initializing')
    net.initialize()
net.to(device)
loss_function = nn.MSELoss()
loss_function.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
loss_eval = 999999999999
# Writer
writer = SummaryWriter("./runs/transfer_eval")
for epoch in range(opt.n_epochs):
    # Train
    net.train()
    print('Training')
    for i, (image_pan, image_ms, image_pan_label, image_ms_label) in enumerate(train_loader):
        # Train
        optimizer.zero_grad()
        image_transfer = net(image_ms_label.to(device))
        loss = loss_function(image_transfer.to(device), image_pan.to(device))
        loss.backward()
        optimizer.step()
        print('Epoch : {}/{}   Batch : {}/{}  Loss : {}'.format(epoch, opt.n_epochs, i, len(train_loader)
                                                                             , loss.item()))

    torch.save(net.state_dict(), './Model/STNet/{}'.format(epoch))
    # Evualtion
    print('Evalution')
    net.eval()
    loss_valid = 0
    for i in range(len(valid_dataset)):
        with torch.no_grad():
            image_pan, image_ms, image_pan_label, image_ms_label = valid_dataset[i]
            image_transfer = net(image_ms_label.unsqueeze(0).to(device))
            loss_valid = loss_valid + loss_function(image_transfer.to(device),image_pan.unsqueeze(0).to(device)).item()
            pass
        pass
    print('Eval Loss : ',loss_valid)
    if loss_valid < loss_eval:
        loss_eval = loss_valid
        torch.save(net.state_dict(), './Model/STNet/transfer.pth'.format(epoch))
        pass
    writer.add_scalar('Eval Loss',loss_valid,global_step=epoch)
    pass
