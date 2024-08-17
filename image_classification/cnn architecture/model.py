import torch.nn as nn
import torch 
from torchsummary import summary

def conv_block(channel_in , channel_out ,kernel_size=3 , stride=1 ,padding=0)->torch.nn:
    return nn.Sequential(
        nn.Conv2d(channel_in,channel_out,kernel_size,stride,[padding]),
        nn.ReLU(),
        nn.BatchNorm2d(channel_out),
        nn.MaxPool2d(2)
    )

def get_model(input_shape,device:torch.device=torch.device('cuda')):
    model= nn.Sequential(
        conv_block(input_shape[0],32),
        conv_block(32,64),
        conv_block(64,128),
        conv_block(128,512),
        conv_block(512,512),
        conv_block(512,512),
        nn.Flatten(),
        nn.Linear(512,1),
        nn.Sigmoid()
    ).to(device)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters())

    return model , loss , optimizer


if __name__ =="__main__": 
    input_shape=(3,224,224)
    model , _, _ = get_model(input_shape)
    print(summary(model,input_shape))