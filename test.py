import torch
import ignite
import matplotlib
import cv2

from image_classification.data import test_dataset
print("PyTorch version:",cv2.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

print('test : ',len(test_dataset))