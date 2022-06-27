import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

#   Channel Attention Block 
class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelAttentionBlock, self).__init__()
        self.reduction = reduction
        self.dct_layer = nn.AdaptiveAvgPool2d(1)# DCTLayer(channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        n,c,h,w = x.size()
        y = self.dct_layer(x).squeeze(-1).squeeze(-1)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


#   Spatial Attention Block
class SpatialAttentionBlock(nn.Module):
    def __init__(self,channel):
        super(SpatialAttentionBlock, self).__init__()
         # Maximum pooling
        self.featureMap_max = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 1),padding=0)
        )
        # Average pooling
        self.featureMap_avg = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.AvgPool2d(kernel_size=(5, 5), stride=(1,1), padding=0)
        )

        # Deviation pooling
        # var = \sqrt(featureMap - featureMap_avg)^2

        # Dimensionality Reduction
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(in_channels=channel * 4, out_channels=channel, kernel_size=(3,3), stride=(1, 1), padding=1,bias=False),
            nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=(1,1),stride=(1,1),bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        x_max = self.featureMap_max(x)
        x_avg = self.featureMap_avg(x)
        x_var = torch.sqrt(torch.pow(x - x_avg,2) + 1e-7)

        y = torch.cat([x_max,x_avg,x_var,x],dim=1)
        z = self.reduce_dim(y)
        return x * z


#   Multi-scale Feature Extract Contains Two Part:
#   Deep Feature Extract Block
class DeepFeatureExtractBlock(nn.Module):
    def __init__(self,in_ch,out_ch,ksize):
        super(DeepFeatureExtractBlock, self).__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.initial_kernel_size = ksize
        self.sub_channels = out_ch//4
        self.DeepFeature_Layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch,kernel_size=(ksize, ksize), stride=(1, 1), padding=ksize//2),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.DeepFeature_Layer2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4,kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.DeepFeature_Layer3 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4,kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.DeepFeature_Layer4 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4,kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.DeepFeature_Layer5 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4,kernel_size=(7, 7), stride=(1, 1), padding=3),
            nn.LeakyReLU(0.2,inplace=True)
        )

    def forward(self,x):
        x_deepfeature = self.DeepFeature_Layer1(x)
        f1,f3,f5,f7 = x_deepfeature[:,0:self.sub_channels,:,:],x_deepfeature[:,self.sub_channels:self.sub_channels*2,:,:]\
        ,x_deepfeature[:,self.sub_channels*2:self.sub_channels*3,:,:],x_deepfeature[:,self.sub_channels*3:self.sub_channels*4,:,:]
        x_deepfeature1 = self.DeepFeature_Layer2(f1)
        x_deepfeature3 = self.DeepFeature_Layer3(f3)
        x_deepfeature5 = self.DeepFeature_Layer4(f5)
        x_deepfeature7 = self.DeepFeature_Layer5(f7)
        x_deepfeature_cat = torch.cat([x_deepfeature1,x_deepfeature3,x_deepfeature5,x_deepfeature7],dim=1)
        y = x_deepfeature + x_deepfeature_cat
        return y


#   Shallow Feature Extract Block
class ShallowFeatureExtractBlock(nn.Module):
    def __init__(self,in_ch,out_ch,reduction = 4):
        super(ShallowFeatureExtractBlock, self).__init__()
        self.ShallowFeature_Layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=(9,9),stride=(1,1),padding=4),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.ShallowFeature_Layer2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch//reduction,kernel_size=(1,1),stride=(1,1),padding=0),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.ShallowFeature_Layer3 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch//reduction,out_channels=out_ch,kernel_size=(5,5),stride=(1,1),padding=2),
            nn.LeakyReLU(0.2,inplace=True)
        )

    def forward(self,x):
        x_shallowfeature_1 = self.ShallowFeature_Layer1(x)
        x_shallowfeature_2 = self.ShallowFeature_Layer2(x_shallowfeature_1)
        x_shallowfeature_3 = self.ShallowFeature_Layer3(x_shallowfeature_2)
        y = x_shallowfeature_1 + x_shallowfeature_3
        return y


#   Multi-scale Feature Extract
class MultiscaleFeatureExtractBlock(nn.Module):
    def __init__(self,channel,ratio):
        super(MultiscaleFeatureExtractBlock, self).__init__()
        # Deep Feature
        self.out_channels = channel * ratio
        self.deepfeature_model1 = DeepFeatureExtractBlock(in_ch=channel,out_ch=self.out_channels,ksize=7)
        self.deepfeature_model2 = DeepFeatureExtractBlock(in_ch=self.out_channels,out_ch=self.out_channels,ksize=3)
        self.deepfeature_model3 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels,out_channels=self.out_channels,kernel_size=(5,5),stride=(1,1),padding=2),
            nn.LeakyReLU(0.2)
        )
        # Shallow Feature
        self.shallowfeature_model = ShallowFeatureExtractBlock(in_ch=channel,out_ch=self.out_channels)

    def forward(self,x):
        df = self.deepfeature_model3(
            self.deepfeature_model2(
                self.deepfeature_model1(x)
            )
        )
        sf = self.shallowfeature_model(x)
        y = torch.cat([df,sf],dim=1)
        return y


if __name__ == '__main__':
    print('Hello world')