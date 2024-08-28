from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from PIL import Image

import torch
from torch import Tensor
from torchvision.transforms import transforms

import cv2
import numpy as np

import typing as tt
# Paths 
script_dir = Path(__file__).parent.resolve()
data_path = (script_dir / '../data').resolve()

train_csv_path = data_path / 'train.csv'
test_csv_path = data_path / 'test.csv'

imgs_path = data_path / 'images'
labels_path = data_path / 'labels'

# Pascal VoC dataset : https://www.kaggle.com/datasets/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2

class PascalDataset(Dataset):
    def __init__(self,csv_path:Path,imgs_path:Path,labels_path:Path,transform,grid_cells:int=7,device='cuda'):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.imgs_path=imgs_path
        self.labels_path=labels_path
        self.grid_cells =grid_cells
        self.NB_CLASSES = 20
        self.device=device

    def __len__(self):
        return len(self.df)
    
    def _get_file_path(self,file_name:str,path:Path):
        for file_path in path.glob(file_name):
            return file_path
        raise FileNotFoundError(f"No files matching '{file_name}' found in directory '{path}'")
    
    def _construct_target_tensor(self,boxes)->Tensor:
        target_matrix = torch.zeros(self.grid_cells,self.grid_cells,self.NB_CLASSES+5)
        for box in boxes:
            c,x,y,w,h=box
            c = int(c)
            # get (i,j) based on x,y ,
            i,j = int(self.grid_cells*y),int(self.grid_cells*x)
            relative_x , relative_y = self.grid_cells*x-j ,self.grid_cells*y-i
            relative_w , relative_h = self.grid_cells*w , self.grid_cells*h
            if target_matrix[i,j,20]==0:
                target_matrix[i,j,20]=1
                target_matrix[i,j,c] =1.0
                target_matrix[i,j,21:25] = torch.tensor([relative_x , relative_y,relative_w , relative_h])
        return target_matrix



    def __getitem__(self,index)->tt.Tuple[Tensor,Tensor]:
        img_file_name , labels_file_name = self.df.iloc[index]
        img_path ,labels_path= self._get_file_path(img_file_name,self.imgs_path),\
                                self._get_file_path(labels_file_name,self.labels_path)
        # The image
        img = Image.open(img_path)
        img_t = self.transform(img)

        boxes =[]
        
        with open(labels_path) as labels_file:
            for line in labels_file.readlines():
                c , x ,y , w ,h = [
                float(nb) if float(nb)!=int(float(nb)) else int(nb)
                for nb in line.split()
                ]
                boxes.append([c,x,y,w,h])
        target_matrix = self._construct_target_tensor(boxes)

        return img_t.to(self.device),target_matrix.to(self.device)

transform = transforms.Compose([
    transforms.Resize((448,448)),
    transforms.ToTensor()
])
        
train_ds = PascalDataset(
    csv_path=train_csv_path,
    imgs_path=imgs_path,
    labels_path=labels_path,
    transform=transform,
    grid_cells=7,
)

valid_ds = PascalDataset(
    csv_path=train_csv_path,
    imgs_path=imgs_path,
    labels_path=labels_path,
    transform=transform,
    grid_cells=7,
)



if __name__ == "__main__":
    img_t, target_matrix = train_ds[1]

    # Convert tensor to numpy array and ensure it's uint8 for OpenCV
    img = img_t.cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    
    # Convert from RGB to BGR for OpenCV
    img = img[:, :, ::-1]

    img = np.ascontiguousarray(img)

    indices = target_matrix[..., 20] != 0
    cells =torch.nonzero(indices).cpu().numpy() # (nb_cells,2)
    boxes_t = target_matrix[indices]



    if boxes_t.numel() == 0:
        print("No objects detected in this image.")
    else:
        boxes = boxes_t.cpu().numpy()
        # Draw the bounding boxes
        for box,cell in zip(boxes,cells):
            i,j = cell
            print(i,j)
            box = box[21:25] 
            print(box)
            x_center, y_center, width, height = box
            x_center = 448* (x_center+j)/7
            y_center =448* (y_center+i)/7
            width =448* width/7
            height= 448*height/7
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        # Display the image with bounding boxes
        cv2.imshow('Image with Bounding Boxes', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
