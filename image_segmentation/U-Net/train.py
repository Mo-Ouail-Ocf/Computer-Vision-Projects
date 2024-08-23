from ignite.engine import Engine 
from utils import attach_ignite
from model import get_model
import torch
from data import train_dl


NUM_CLASSES = 14

unet_model , loss_fn , optimizer = get_model(NUM_CLASSES,'cuda')

def train_step(engine,batch):
    unet_model.train()
    unet_model.zero_grad()
    imgs , targets = batch # tagets shape : (batch_size ,w,h)
    predictions = unet_model(imgs) # shape : (batch_size , num_classes , w , h)
    loss = loss_fn(predictions,targets)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad
def valid_step(engine,batch):
    unet_model.eval()
    imgs , targets = batch # tagets shape : (batch_size ,w,h)
    predictions = unet_model(imgs) # shape : (batch_size , num_classes , w , h)
    return predictions,targets

trainer = Engine(train_step)
evaluator = Engine(valid_step)

attach_ignite(trainer,evaluator,unet_model,loss_fn)


if __name__=="__main__":
    trainer.run(train_dl,max_epochs=15)