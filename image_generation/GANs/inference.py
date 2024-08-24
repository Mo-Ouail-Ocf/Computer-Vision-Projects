import pathlib
import torch
from model import Generator
import matplotlib.pyplot as plt

curr_path = pathlib.Path(__file__).parent
model_path = curr_path / 'checkpoint/model.pt'

model = Generator().to('cuda')

model.load_state_dict(torch.load(model_path, weights_only=True))

model.eval()

if __name__ == "__main__":
    with torch.no_grad():
        z = torch.randn(1, 100, 1, 1).to('cuda')  
        img = model(z)  
        img_cpu = img.cpu().squeeze(0) 
        img_cpu = (img_cpu + 1) / 2 

        img_np = img_cpu.permute(1, 2, 0).numpy()  
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.imshow(img_np)
        ax.axis('off')  # Hide axes
        plt.show()
