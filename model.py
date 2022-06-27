from modules import *


# Fusion Network
class FusionNetwork(nn.Module):
    def __init__(self,pan_channel = 1,ms_channel = 4,ratio = 8):
        super(FusionNetwork, self).__init__()
        # Panchromatic
        self.pan_feature_extract_model = MultiscaleFeatureExtractBlock(channel=pan_channel,ratio = ratio)
        self.pan_channel_attention_model = ChannelAttentionBlock(channel=pan_channel * ratio * 2 + 1)
        self.pan_spatial_attention_model = SpatialAttentionBlock(channel=pan_channel * ratio * 2 + 1)
        # Multi-Scale
        self.ms_feature_extract_model = MultiscaleFeatureExtractBlock(channel=ms_channel,ratio = ratio)
        self.ms_channel_attention_model = ChannelAttentionBlock(channel=ms_channel * ratio * 2 + 4)
        self.ms_spatial_attention_model = SpatialAttentionBlock(channel=ms_channel * ratio * 2 + 4)
        # Feature Map
        self.feature_channel_attention_model = ChannelAttentionBlock(channel= pan_channel + ms_channel
                                + 2 * (pan_channel * ratio * 2 + 1) + 2 * (ms_channel * ratio * 2 + 4))
        self.feature_spatial_attention_model = SpatialAttentionBlock(channel= pan_channel + ms_channel
                                + 2 * (pan_channel * ratio * 2 + 1) + 2 * (ms_channel * ratio * 2 + 4))
        # Fusion
        self.fusion_model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels= 2 * (pan_channel + ms_channel
                                + 2 * (pan_channel * ratio * 2 + 1) + 2 * (ms_channel * ratio * 2 + 4)),
                      out_channels = 128,kernel_size=(3,3),stride=(1,1),padding=0,bias=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=0,bias=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True),
            nn.Tanh()
        )

    def initialize(self):
        for model in self.modules():
            if isinstance(model, nn.Conv2d):
                nn.init.trunc_normal_(model.weight, mean=0.0, std=1e-3)
                if model.bias is not None:
                    nn.init.constant_(model.bias, val=0.0)

    def forward(self,pan,ms):
        # Panchromatic
        #   Feature Extract
        pan_features = self.pan_feature_extract_model(pan)
        pan_in_features = torch.cat([pan,pan_features],dim=1) 
        #   Attention Mechanism
        pan_ca_features = self.pan_channel_attention_model(pan_in_features) 
        pan_sa_features = self.pan_spatial_attention_model(pan_in_features) 
        pan_out_features = torch.cat([pan_ca_features,pan_sa_features],dim=1)

        # Multi-Spectral
        #   Feature Extract
        n,c,h,w = ms.size()
        ms_up = F.interpolate(ms, size=(4 * h, 4 * w), mode='bicubic', align_corners=False)
        ms_features = self.ms_feature_extract_model(ms_up) 
        ms_in_features = torch.cat([ms_up, ms_features], dim=1) 
        #   Attention Mechanism
        ms_ca_features = self.ms_channel_attention_model(ms_in_features)
        ms_sa_features = self.ms_spatial_attention_model(ms_in_features)

        ms_out_features = torch.cat([ms_ca_features,ms_sa_features],dim=1)

        # Concatenate 
        features_cat = torch.cat([pan,pan_out_features,ms_up,ms_out_features],dim=1)
        features_cat_ca = self.feature_channel_attention_model(features_cat)
        features_cat_sa = self.feature_spatial_attention_model(features_cat)
        # Fusion
        features_input = torch.cat([features_cat_ca,features_cat_sa],dim=1)
        features_output = self.fusion_model(features_input) + ms_up
        return features_output


# Transfer Network Archtecture
class TransferNetwork(nn.Module):
    def __init__(self):
        super(TransferNetwork, self).__init__()
        self.Layer1 = nn.Sequential(
            # Layer1
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.Layer2 = nn.Sequential(
            # Layer2
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.Layer3 = nn.Sequential(
            # Layer3
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Concatanate
        self.Layer4 = nn.Sequential(
            # Layer4
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Concatanate
        self.Layer5 = nn.Sequential(
            # Layer5
            nn.Conv2d(in_channels=20, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.Layer6 = nn.Sequential(
            # Layer6
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.Tanh()
        )

    def initialize(self):
        for model in self.modules():
            if isinstance(model, nn.Conv2d):
                nn.init.trunc_normal_(model.weight, mean=0.0, std=1e-3)
                nn.init.constant_(model.bias, val=0.0)
    
    def forward(self, x):
        y1 = self.Layer1(x)
        y2 = self.Layer2(y1)
        y3 = self.Layer3(y2)
        x1 = torch.cat([y1, y3], dim=1)
        y4 = self.Layer4(x1)
        x2 = torch.cat([x, y4], dim=1)
        y5 = self.Layer5(x2)
        y6 = self.Layer6(y5)
        return y6