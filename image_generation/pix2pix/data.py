from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config
import numpy as np
data_path = Path(__file__).parent / 'pix2pix_dataset' / 'maps' / 'maps' 

train_path = data_path / 'train'
val_path = data_path / 'val'

def get_imgs_files(path:Path)->list[str]:
    img_files = list(path.glob('*.jpg'))
    img_files = [str(img_file) for img_file in img_files]
    return img_files
    
class MapDataset(Dataset):
    def __init__(self,path:Path,tranform,valid=False,device='cuda'):
        super().__init__()
        self.files_paths = get_imgs_files(path)
        self.transform = tranform
        self.device=device
        if valid:
            self.files_paths = self.files_paths[:16] # we dont need to much data for validation logging 
    
    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, index) :
        img_path = self.files_paths[index] 
        full_img = Image.open(img_path) 
        full_img = np.array(full_img)
        input_img  , output_img = full_img[:,:600,:],full_img[:,600:,:]

        transformed = self.transform(image=input_img, output_image=output_img)
        input_img_t = transformed['image']
        output_img_t = transformed['output_image']
    
        return input_img_t.to(self.device), output_img_t.to(self.device)
    

mean ,std= [0.5,0.5,0.5],[0.5,0.5,0.5]
transform = A.Compose(
    [
        A.Resize(256,256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean,std=std,max_pixel_value=255.0),
        ToTensorV2()
    ] ,
    additional_targets= {
        'output_image':'image'
    }
)


train_ds = MapDataset(train_path,tranform=transform)
val_ds = MapDataset(val_path,valid=True,tranform=transform)

train_dl = DataLoader(
    dataset=train_ds,
    batch_size=Config.BATCH_SIZE,
    drop_last=True,
    num_workers=Config.NUM_WORKERS,
    shuffle=True
)

valid_dl = DataLoader(
    dataset=val_ds,
    batch_size=Config.BATCH_SIZE,
    drop_last=True,
    num_workers=Config.NUM_WORKERS,
    shuffle=True
)
