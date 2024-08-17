import torch.nn as nn
import torch 
from torchsummary import summary
from torchvision import  models


def get_model(device:torch.device=torch.device('cuda')):
    model = models.vgg16()
    for param in model.parameters():
        param.requires_grad =  False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128,1),
        nn.Sigmoid()
    )
    model=model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters())
    return model , loss_fn , optimizer


if __name__=="__main__":
    model,_,_ = get_model()
    print(summary(model,(3,224,224)))