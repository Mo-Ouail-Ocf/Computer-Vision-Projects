import torch
from torch import nn
import warnings


from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.handlers.tensorboard_logger import TensorboardLogger,OutputHandler

def attach_ignite(
        trainer:Engine,
        model:nn.Module
    ):


    @trainer.on(Events.EPOCH_COMPLETED)
    def log(engine):
        output=trainer.state.output
        loss_gen = output['loss_gen']
        loss_disc = output['loss_disc']
        print(f'~~Epoch {trainer.state.epoch} completed ~~~~')
        print(f'-> Gen loss : {loss_gen}')
        print(f'-> Disc loss : {loss_disc}')
    
    pb_bar = ProgressBar()
    pb_bar.attach(trainer, 
                  output_transform=lambda x: 
                  {
                      'loss_gen': x['loss_gen'],
                      'loss_disc':x['loss_disc']
                   },
                )
    
    output_handler =OutputHandler(
        output_transform=lambda x: 
                  {
                      'loss_gen': x['loss_gen'],
                      'loss_disc':x['loss_disc']
                   },
        tag="train"
    )

    tb_logger = TensorboardLogger(log_dir="gan")

    tb_logger.attach(
        trainer,
        output_handler,
        event_name=Events.EPOCH_COMPLETED
    )

    def score_function(engine:Engine):
        return -engine.state.output['loss_gen']

    model_checkpoint = ModelCheckpoint(
    "checkpoint",
    n_saved=1,
    filename_prefix="best_gan",
    score_function=score_function,
    score_name="loss_gen",
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, model_checkpoint, {"model": model}
    )
    