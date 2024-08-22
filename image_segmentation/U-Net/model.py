from torchvision.models.vgg import vgg16_bn
import torch.nn as nn
import torch


def conv(in_features:int , out_features:int,nb_blocks=3)->nn.Module:
    block = nn.Sequential()

    for i in range(nb_blocks):
        block.append(nn.Conv2d(in_features,out_features,3,1,1))
        block.append(nn.BatchNorm2d(out_features))
        block.append(nn.ReLU())
        in_features=out_features
    return block



def up_conv(in_features:int , out_features:int)->nn.Module:
    return nn.Sequential(
        nn.ConvTranspose2d(in_features,out_features,kernel_size=2,stride=2),
        nn.ReLU()
    )

class UNet(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.encoder = vgg16_bn(pretrained=True).features
        
        self.block1 = nn.Sequential(*self.encoder[0:6]) # 64 : 224 
        self.block2 = nn.Sequential(*self.encoder[6:13]) # 128 : 112
        self.block3 = nn.Sequential(*self.encoder[13:23]) # 256 : 56
        self.block4 = nn.Sequential(*self.encoder[23:33]) # 512 : 28

        self.bottleneck = nn.Sequential(nn.MaxPool2d(2),*conv(512,1024,3))

        self.up_conv4 = up_conv(1024,512) # 28  
        self.conv4 = conv(512+512,512)
        
        self.up_conv3 = up_conv(512,256)
        self.conv3 = conv(512,256)
        
        self.up_conv2 = up_conv(256,128)
        self.conv2 = conv(256,128)

        self.up_conv1 = up_conv(128,64)
        self.conv1 = conv(128,64)

        self.output_layer = nn.Conv2d(64,n_classes,3,1,1)

    def forward(self,x):
        x1_encoder= self.block1(x) 
        x2_encoder= self.block2(x1_encoder)
        x3_encoder= self.block3(x2_encoder)
        x4_encoder= self.block4(x3_encoder)
        x5_encoder=self.bottleneck(x4_encoder)

        # bottleneck transition
        x5 = self.up_conv4(x5_encoder)
        x4_decoder = torch.cat([x5,x4_encoder],dim=1)
        x4_decoder = self.conv4(x4_decoder)
        x3_decoder = self.up_conv3(x4_decoder)

        # decoder path : stage 3
        x3_decoder=torch.cat([x3_decoder,x3_encoder],dim=1)
        x3_decoder=self.conv3(x3_decoder)
        x2_decoder = self.up_conv2(x3_decoder)

        # decoder path : stage 2
        x2_decoder=torch.cat([x2_decoder,x2_encoder],dim=1)
        x2_decoder=self.conv2(x2_decoder) 
        x1_decoder = self.up_conv1(x2_decoder)

        # decoder path : stage 1
        x1_decoder=torch.cat([x1_decoder,x1_encoder],dim=1)
        x1_decoder=self.conv1(x1_decoder)
        
        
        out = self.output_layer(x1_decoder)

        return out
    

def get_model(num_classes,device='cuda')->tuple[nn.Module,nn.CrossEntropyLoss,torch.optim.Adam]:
    model = UNet(num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters())
    return model ,loss_fn, optim

