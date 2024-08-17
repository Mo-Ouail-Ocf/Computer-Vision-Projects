from data import train_dl , valid_dl
from model import get_model
from ignite.engine import create_supervised_evaluator , create_supervised_trainer ,Events,Engine
from ignite.contrib.handlers.tensorboard_logger import OutputHandler,TensorboardLogger
from ignite.metrics import Accuracy,Loss
from ignite.handlers import Checkpoint, DiskSaver
from ignite.handlers import EarlyStopping
from ignite.handlers.tqdm_logger import ProgressBar
from utilities import log_images_labels

input_shape = (3,224,224)

model , loss_fn , optimizer = get_model()

# Models

trainer = create_supervised_trainer(model,optimizer,loss_fn=loss_fn,device="cuda")

def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = (y_pred > 0.5).float()  # Convert probabilities to 0 or 1 based on a 0.5 threshold
    return y_pred, y

evaluator = create_supervised_evaluator(model,
    metrics={
    'accuracy':Accuracy(output_transform=thresholded_output_transform),
    'loss':Loss(loss_fn)
    },
    device="cuda")

# events
pbar = ProgressBar()
pbar.attach(trainer, output_transform=lambda x: {'loss': x})



@trainer.on(Events.EPOCH_COMPLETED)
def log_valid_metrics(engine):
    eval_state = evaluator.run(valid_dl)
    metrics = eval_state.metrics
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f"---> Epoch  {trainer.state.epoch} completed \n")
    print(f"-----> Train Loss : {trainer.state.output} \n")
    print(f"-----> Validation Loss : {metrics['loss']} \n")
    print(f"-----> Validation Accuracy: {metrics['accuracy']} \n")
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    log_images_labels(trainer,model,tb_logger,valid_dl)

# Logging
tb_logger =TensorboardLogger(log_dir='./logs')

# valid
metrics_logger = OutputHandler(metric_names=['accuracy','loss'],tag="validation")
tb_logger.attach(
    evaluator,
    metrics_logger,
    event_name=Events.EPOCH_COMPLETED,
)

#train
train_loss_logger = OutputHandler(
    output_transform= lambda loss : {"loss":loss},tag="train")
tb_logger.attach(
    trainer,
    train_loss_logger,
    event_name=Events.EPOCH_COMPLETED,
)

# early stopping
def score_function(engine):
    val_loss = engine.state.metrics['loss']
    return -val_loss
handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)

trainer.run(train_dl,max_epochs=20)
