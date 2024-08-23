import pathlib
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader ,Dataset
from torchvision.transforms import transforms
import typing as tt
import torch
import numpy as np 
import matplotlib.pyplot as plt
CURR_PATH = pathlib.Path(__file__).parent
DATA_PATH = CURR_PATH / 'dataset1'
TRAIN_PATH_TARGET = DATA_PATH / 'annotations_prepped_train'
TRAIN_PATH_DATA = DATA_PATH / 'images_prepped_train'

VALID_PATH_TARGET = DATA_PATH / 'annotations_prepped_test'
VALID_PATH_DATA = DATA_PATH / 'images_prepped_test' 


def get_data_target_files(data_path:Path,target_path:Path):
    data_files = [ str(f) for f in data_path.glob('*.png')]
    target_files = [ str(f) for f in target_path.glob('*.png')]
    return data_files , target_files


class UNetDataSet(Dataset):
    def __init__(self,data_files:list[str], target_files:list[str],transform:transforms.Compose,transform_target:transforms.Compose,device='cuda'):
        self.transform_data = transform
        self.transform_target=transform_target
        self.data_files = data_files
        self.target_files =  target_files
        self.device = device

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index)->tt.Tuple[torch.Tensor,torch.Tensor]:
        data_file , target_file = self.data_files[index],self.target_files[index]
        img , target_img = Image.open(data_file).resize((224,224)) , Image.open(target_file).resize((224,224))
        img_t , target_img_t = self.transform_data(img),torch.LongTensor(np.array(target_img)) # size : (3,h,w) ; (h,w)
        return img_t.to(self.device ),target_img_t.to(self.device)


mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
transform_target =transforms.Compose([
            transforms.ToTensor()
        ])
unnormalize = transforms.Compose([
    transforms.Normalize(mean=-mean / std, std=1 / std)
])


test_data_files , test_target_files = get_data_target_files(VALID_PATH_DATA,VALID_PATH_TARGET) 
train_data_files , train_target_files = get_data_target_files(TRAIN_PATH_DATA,TRAIN_PATH_TARGET) 


valid_dataset = UNetDataSet(test_data_files , test_target_files,transform,transform_target)
train_dataset = UNetDataSet(train_data_files , train_target_files,transform,transform_target)

valid_dl = DataLoader(valid_dataset,shuffle=True,batch_size=4)
train_dl = DataLoader(train_dataset,shuffle=True,batch_size=1)


if __name__=="__main__":
    for i in range(len(train_dataset)):
        img1 , img2 = train_dataset[i] 
        print(img2.min())
    
    """ fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Display the first image
    axes[0].imshow(unnormalize(img1).cpu().permute(1,2,0))  # Convert CHW to HWC for display
    axes[0].set_title('Image 1')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Display the second image
    cbar_img = axes[1].imshow(img2.squeeze(0).cpu(), cmap='jet')  # Single-channel image
    axes[1].set_title('Image 2')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Add a colorbar for the second image
    fig.colorbar(cbar_img, ax=axes[1]) """

    #plt.show()

 
