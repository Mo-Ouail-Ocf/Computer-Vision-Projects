from model import get_model
from data import  train_dl
from utils import attach_ignite
import torch
from torch import nn

from ignite.engine import Engine

# Model & Evaluator :
model , loss_age , loss_gender , optimizer = get_model()


def train_step(engine,batch):
    model.train()
    optimizer.zero_grad()
    imgs_t , ages_t , genders_t = batch[0] , batch[1] , batch[2]
    age_preds , gender_probas = model(imgs_t)
    ages_loss = loss_age(age_preds,ages_t)
    genders_classif_loss = loss_gender(gender_probas,genders_t)
    loss = ages_loss+genders_classif_loss
    loss.backward()
    optimizer.step()
    return loss.item()

def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        imgs_t , ages_t , genders_t = batch[0] , batch[1] , batch[2]
        age_preds , gender_probas = model(imgs_t)
        return age_preds,ages_t,gender_probas,genders_t
    

trainer = Engine(train_step)
evaluator = Engine(validation_step)


if __name__=="__main__":
    attach_ignite(
        trainer,
        evaluator,
        model,
        loss_age
    )
    trainer.run(train_dl,max_epochs=15)
