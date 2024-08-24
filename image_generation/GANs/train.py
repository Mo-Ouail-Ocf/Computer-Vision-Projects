import torch
from model import get_gen_disc,disc_train_step,gen_train_step,get_synth_images
from data import face_dl
from ignite.engine import Engine
from utils import attach_ignite
import warnings

def train():
    model_gen,optim_gen,model_disc,optim_disc,loss_fn = get_gen_disc()

    def process_batch(engine,real_data_imgs:torch.Tensor):
        # Fix Generator , Improve disciriminator 
        model_disc.zero_grad()
        batch_size = real_data_imgs.shape[0]
        synthethic_imgs=get_synth_images(model_gen,batch_size,False)
        loss_disc = disc_train_step(model_disc,loss_fn,real_data_imgs,synthethic_imgs)
        loss_disc.backward()
        optim_disc.step()

        # Improve Generator , Fix disciriminator
        model_gen.zero_grad()
        loss_gen = gen_train_step(model_gen,model_disc,batch_size,loss_fn)
        loss_gen.backward()
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        warnings.simplefilter(action='ignore', category=DeprecationWarning)


        return {
            'loss_gen':loss_gen.item(),
            'loss_disc':loss_disc.item()
        }

    trainer = Engine(process_batch)

    attach_ignite(trainer,model_gen)

    trainer.run(face_dl,max_epochs=15)


if __name__=="__main__":
    train()
    

    
