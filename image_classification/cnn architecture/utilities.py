from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from ignite.engine import Events,Engine
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from torchvision.utils import make_grid

import torch.nn.functional as F
# logging images
to_pil = transforms.ToPILImage()

def log_images_labels(engine: Engine,
                      model: nn.Module,
                      tb_logger: TensorboardLogger,
                      validation_dl: DataLoader,
                      num_imgs=5):
    model.eval()
    imgs, labels = next(iter(validation_dl))
    imgs, labels = imgs[:num_imgs], labels[:num_imgs]

    with torch.no_grad():
        logits = model(imgs)
        predictions = torch.sigmoid(logits)
        rounded_predictions = torch.round(predictions)


    # Log the grids to TensorBoard

    for img , label , pred in zip(imgs,labels,rounded_predictions): 
        tb_logger.writer.add_image(f'Predicted class : {pred}', img, engine.state.epoch)

# Attach this function to the engine
""" @engine.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    log_images_labels(engine, model, tb_logger, valid_dl) """
