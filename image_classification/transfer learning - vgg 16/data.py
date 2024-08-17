from torch.utils.data import Dataset , DataLoader
from glob import glob
from random import shuffle, seed
import os
import pathlib
import cv2
import torch
from torchvision import transforms
seed(10)

class CatsDogsDataSet(Dataset):
    def __init__(self, root_path, path,device:torch.device=torch.device('cuda')):
        super().__init__()  
        self.data_path = root_path / path
        
        # print(f"Looking for images in: {self.data_path.resolve()}")

        cats_path = glob(os.path.join(self.data_path, "cats", "*.jpg"))
        dogs_path = glob(os.path.join(self.data_path, "dogs", "*.jpg"))
        self.fpath = cats_path + dogs_path
        # Shuffle
        shuffle(self.fpath)
        self.targets = [file_name.split('\\')[-1].startswith('dog') for file_name in self.fpath]
        self.device=device
        # same normalization as VGG16 pretrain data
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224,  0.225]
        )

    def __len__(self):
        return len(self.fpath)

    def __getitem__(self, index):
        img_path = self.fpath[index]
        # Convert to RGB to ensure consistency
        img = cv2.imread(img_path)[:,:,::-1]
        img = cv2.resize(img,(224,224))
        img_t = torch.tensor(img/255).permute(2,0,1).float()
        img_t = self.normalize(img_t).to(self.device)
        target = self.targets[index]
        target_t = torch.tensor([target],dtype=torch.float).to(self.device)
        return img_t , target_t

# Determine the root path dynamically based on the script's location
current_script_dir = pathlib.Path(__file__).parent
root_path = current_script_dir / 'cat-and-dog'

# Create datasets
test_dataset = CatsDogsDataSet(root_path, 'test_set/test_set')
train_dataset = CatsDogsDataSet(root_path, 'training_set/training_set')

# data loaders

train_dl = DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=True)
valid_dl = DataLoader(test_dataset,batch_size=32,shuffle=True,drop_last=True)



 