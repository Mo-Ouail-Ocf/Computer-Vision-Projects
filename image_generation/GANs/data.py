import pathlib
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import typing as tt
import torch
from PIL import Image
CURR_PATH = pathlib.Path(__file__).parent


DATA_PATH_F = CURR_PATH / 'females'
DATA_PATH_M = CURR_PATH / 'males'
CROPPED_IMGS_PATH = CURR_PATH / 'cropped'

def create_cropped_dataset(cropped_imgs_path: pathlib.Path):

    cropped_files : tt.List[str] = []
    cropped_imgs_path.mkdir(parents=True, exist_ok=True)

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    f_files = [f for f in DATA_PATH_F.glob('*.jpg')]
    m_files = [f for f in DATA_PATH_M.glob('*.jpg')]
    files = f_files + m_files
    
    for i, file_path in enumerate(files):
        if i == 100:  # Limit to 100 images
            break
        
        img = cv2.imread(str(file_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for j, (x, y, w, h) in enumerate(faces):
            cropped_img = img[y:y+h, x:x+w]
            output_path = cropped_imgs_path / f'{i}_{j}.jpg'
            cv2.imwrite(str(output_path), cropped_img)
            cropped_files.append(str(output_path))
    return cropped_files

cropped_files=create_cropped_dataset(CROPPED_IMGS_PATH)

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)
    ),
])

'''
Considerations for the Order
-Spatial Transformations: Operations like resizing, cropping, and flipping should generally be performed before 
 converting the image to a tensor (ToTensor()), as these operations are more intuitive and efficient to perform on PIL images.
-Pixel Value Transformations: Conversions like normalization should occur after the image has been converted to a tensor, as these 
 operations are mathematically defined on tensors.
'''

class FacesDataset(Dataset):
    def __init__(self,transform : transforms.Compose,file_names:tt.List[str],device='cuda'):
        self.transform = transform
        self.file_names = file_names
        self.device=device

    def __len__(self): 
        return len(self.file_names)

    def __getitem__(self, index)->torch.Tensor:
        img_file = self.file_names[index]
        img = Image.open(img_file)
        img_t:torch.Tensor = self.transform(img)
        return img_t.to(self.device)
    

face_dataset = FacesDataset(transform,cropped_files)

face_dl = DataLoader(face_dataset,shuffle=True,batch_size=64,num_workers=4)


if __name__=="__main__":
    print(cropped_files[-1])