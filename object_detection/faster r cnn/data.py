from torch.utils.data import DataLoader , Dataset
from torchsummary import summary
import os
from sklearn.model_selection import train_test_split
import pathlib
from PIL import Image
import pandas as pd
from torchvision.transforms import transforms
import typing as tt
import torch
current_script_dir = pathlib.Path(__file__).parent
DATA_PATH = current_script_dir / 'images/images'
DF_PATH = current_script_dir / 'df.csv'


DF = pd.read_csv(DF_PATH)

def get_imgs_files():
    files = [os.path.join(DATA_PATH,file_name) for file_name in os.listdir(DATA_PATH)]
    return files

def get_test_train_files(files:list[str]):
    train , test = train_test_split(files,test_size=0.2)
    return train,test

def encoding_decoding_lables():
    labels = DF['LabelName'].unique()
    label2target = {
        label:target for target ,label in enumerate(labels)
    }
    target2label ={
        target:label for label , target in label2target.items()
    }
    return label2target , target2label

class BusTruckDataset(Dataset):
    def __init__(self,file_names:list[str],device='cuda',transform=None,label2target:tt.Dict[str,int]={}):
        self.file_paths = file_names
        self.DF = DF
        self.transform = transform
        self.label2target = label2target
        self.device = device

    def __len__(self):  
        return len(self.file_paths)
    
    def __getitem__(self, index) :
        
        
        img_path = str(self.file_paths[index])
        img = Image.open(img_path)
        w ,h = img.size
        img_t = self.transform(img)
        # image infos
        img_id = img_path.split('\\')[-1].split('.')[0]
        print(img_id)
        img_infos = self.DF[ self.DF['ImageID'] == img_id ]
        labels = img_infos['LabelName'].to_list()
        encoded_lables =[self.label2target[label] for label in labels]
        boxes = img_infos[['XMin','YMin','XMax','YMax']].values
        boxes[:,[0,2]]*=224
        boxes[:,[1,3]]*=224

        boxes_t = torch.tensor(boxes,dtype=torch.float32).to(self.device)
        encoded_lable_t = torch.LongTensor(encoded_lables).to(self.device)
        targets ={
            'boxes':boxes_t,
            'labels':encoded_lable_t
        }

        return img_path,img_t , targets



transform = transforms.ToTensor()
label2target , target2label = encoding_decoding_lables()

train_files,test_files = get_test_train_files(get_imgs_files())


test_dataset = BusTruckDataset(file_names=test_files,transform=transform,label2target=label2target)
train_dataset = BusTruckDataset(file_names=train_files,transform=transform,label2target=label2target)

if __name__=="__main__":
    img_path,img_t , targets = next(iter(test_dataset))
    img = Image.open(img_path).resize((224,224), resample=Image.BILINEAR)
    
   


