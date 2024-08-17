from sklearn.model_selection import train_test_split
import os
from torch.utils.data import DataLoader , Dataset
from torchvision.transforms import transforms
from PIL import Image
import torch
DATA_DIR =  'data/utkface/UTKFace'

def get_train_valid_files():
    imgs_paths = [os.path.join(DATA_DIR,file_name) for file_name in os.listdir(DATA_DIR) if file_name.endswith('.jpg')]
    train_imgs_paths , valid_imgs_paths = train_test_split(imgs_paths,test_size=0.2,random_state=42)
    return train_imgs_paths, valid_imgs_paths 

class UTKFace(Dataset):
    def __init__(self,imgs_paths,transform,device:torch.device='cuda'):
        self.imgs_paths = imgs_paths
        self.transform=transform
        self.device = device
    def __len__(self):
        return len(self.imgs_paths)
        
    def __getitem__(self, index):
        img_path = self.imgs_paths[index]
        img = Image.open(img_path)

        img_t = self.transform(img).to(self.device)
        # filenames: [age]_[gender]_[race]_[date&time].jpg
        age = float(img_path.split('_')[0].split('/')[-1])
        age_t = torch.tensor([age],dtype=torch.float).to(self.device)
        gender = float(img_path.split('_')[1])
        gender_t= torch.tensor([gender],dtype=torch.float).to(self.device)
        return img_t , age_t , gender_t

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    # Normalization for VGG16
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

train_imgs_paths, valid_imgs_paths =get_train_valid_files()



train_dataset = UTKFace(train_imgs_paths,transform)
valid_dataset = UTKFace(train_imgs_paths,transform)

train_dl , valid_dl = DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=True) , DataLoader(valid_dataset,batch_size=32,shuffle=True,drop_last=True)

if __name__=="__main__":
    img , age , gender = next(iter(train_dataset))
    print(age)
    print(img.shape)