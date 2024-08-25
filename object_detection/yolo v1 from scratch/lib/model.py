import torch.nn as nn
import torch
import typing as tt
import numpy as np


# Tuple (kernel_size,nb_channels,stride,padding)
# List [Tuples , nb_repeats]
# "M" : max pool layer

ConvLayerTuple = tt.Tuple[int,int,int,int]
# (kernel_size , out_channels , stride , padding)
SeqConvList =tt.List[ConvLayerTuple | int]
# [sequence of conv  , nb repeats]
ConvConfigList = tt.List[ConvLayerTuple| SeqConvList | str  ]
# str : max pooling

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(ConvBlock,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self,x):
        return self.layer(x)


def create_conv_net(in_channels:int,conv_config:ConvConfigList)->nn.Sequential:
    net = nn.Sequential()
    for config in conv_config:
        if isinstance(config,str):
            net.append(nn.MaxPool2d(2,2))
            continue
        if isinstance(config,tuple):
            kernel_size , out_channels , stride , padding = config
            layer = ConvBlock(in_channels,out_channels,kernel_size,stride,padding)
            net.append(layer)
            in_channels = out_channels
        else:
            nb_repeats=config.pop()
            for _ in range(nb_repeats):
                for config_tuple in config:
                    kernel_size , out_channels , stride , padding = config_tuple
                    layer = ConvBlock(in_channels,out_channels,kernel_size,stride,padding)
                    net.append(layer)
                    in_channels = out_channels
    return net

class YoloV1(nn.Module):
    def __init__(self,channels_in,conv_config:ConvConfigList,nb_boxes:int,nb_classes:int):
        super(YoloV1,self).__init__()
        self.conv_blocks = create_conv_net(channels_in,conv_config)
        
        self.out_conv_shape = self.conv_blocks(torch.zeros((1,3,448,448))).shape 
        self.grid = self.out_conv_shape[2] # S
        self.nb_classes = nb_classes # C
        self.nb_boxes = nb_boxes # B : per cell

        out_flatten = np.prod(tuple(self.out_conv_shape))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_flatten,496),
            nn.ReLU(), # original paper : 4096 ; cheating for VRAM :(
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496,(self.grid**2)*(self.nb_classes+self.nb_boxes*5)) # (batch_size , S*S+5*B+C) -> to be reshaped

        )

    def forward(self,x):
        out = self.conv_blocks(x)
        return self.fc(out)

conv_config:ConvConfigList = [
    (7,64,2,3) , # from 448 to 112
    "M",
    (3,192,1,1),
    "M",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),    
    (3,512,1,1),
    "M",
    [(1,256,1,0),(3,512,1,1),4],
    (1,512,1,0),
    (3,1024,1,1),
    "M",
    [(1,512,1,0),(3,1024,1,1),2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1),
]

conv_blocks = YoloV1(3,conv_config,2,20)
out_conv_shape = conv_blocks(torch.zeros((1,3,448,448))).shape



print(out_conv_shape)
