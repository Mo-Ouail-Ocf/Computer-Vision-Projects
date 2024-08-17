import torch
from torch import nn
from data import valid_dl 

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine 
from ignite.contrib.handlers import ProgressBar
 

def attach_ignite(
        trainer:Engine,
        evaluator:Engine,
        model:nn.Module,
        reg_loss : nn.MSELoss 
):
    
    pb_bar = ProgressBar()
    pb_bar.attach(trainer, output_transform=lambda x: {'loss': x})
    # Metrics :
    def accuracy_transf(output):
        _,_,gender_probas,genders_t=output
        gender_pred = torch.round(gender_probas) 
        return gender_pred , genders_t

    def reg_loss_transf(output):
        age_preds,ages_t,_,_ = output
        return age_preds,ages_t
    
    metrics = {
     'reg_loss':Loss(reg_loss,output_transform=reg_loss_transf),
     'classif_accuracy':Accuracy(output_transform=accuracy_transf)
    }

    for name , metric in metrics.items():
       metric.attach(evaluator,name)
   
   # Events :
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        eval_state =evaluator.run(valid_dl)
        metrics = eval_state.metrics
        print("----------\n")
        print(f"> Batch {trainer.state.epoch} \n")
        print(f"> Train loss : {trainer.state.output} \n")
        print(f"> Validation accuracy in classification : {metrics['classif_accuracy']}")
        print(f"> Validation loss in age prediction : {metrics['reg_loss']}")


    # tb logging :
    tb_logger = TensorboardLogger(log_dir="tb-logger")

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="train",
        output_transform=lambda loss: {"batch_loss": loss}
    )

    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="valid",
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

    def score_function(engine):
        return engine.state.metrics["classif_accuracy"]
    
    model_checkpoint = ModelCheckpoint(
    "checkpoint",
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})


    