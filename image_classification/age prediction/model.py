from torchvision import models

import torch
import torch.nn as nn
from torchsummary import summary

class AgeGenderClassifier(nn.Module):
    def __init__(self) :
        super().__init__()
        self.intermediate = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,64),
            nn.ReLU(),
        )  
        self.age_classifier = nn.Sequential(
            nn.Linear(64,1)
        ) 
        self.gender_classifier = nn.Sequential(
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.intermediate(x)
        age = self.age_classifier(x)
        gender_proba = self.gender_classifier(x)
        return age , gender_proba


def get_model(device='cuda'):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad=False

    model.avgpool=nn.Sequential(
        nn.Conv2d(512,512,kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    classifier = AgeGenderClassifier()
    model.classifier = classifier

    model = model.to('cuda')
    optimizer = torch.optim.Adam(params=model.parameters())

    loss_age = nn.MSELoss()
    loss_gender = nn.BCELoss()

    return model , loss_age , loss_gender , optimizer



if __name__=="__main__":
    model ,_,_ ,_= get_model()
    print(summary(model,(3,224,224)))

