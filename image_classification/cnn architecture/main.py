from data import train_dl , valid_dl
from model import get_model
from ignite.engine import create_supervised_evaluator , create_supervised_trainer ,Events,Engine
from ignite.contrib.handlers.tensorboard_logger import OutputHandler,TensorboardLogger
from ignite.metrics import Accuracy,Loss
from utilities import log_images_labels

input_shape = (3,224,224)

model , loss_fn , optimizer = get_model(input_shape)

# Models

trainer = create_supervised_trainer(model,optimizer,loss_fn=loss_fn)

evaluator = create_supervised_evaluator(model,metrics={
    'accuracy':Accuracy(),
    'loss':Loss(loss_fn)
})

# events
@trainer.on(Events.EPOCH_COMPLETED)
def valid(engine):
    evaluator.run(valid_dl)
    print(f"---> Epoch  {trainer.state.epoch} completed")
    print(f"-----> Loss : {trainer.state.output}")
    log_images_labels(trainer,model,tb_logger,valid_dl)

# Logging
tb_logger =TensorboardLogger(log_dir='./logs')

# valid
metrics_logger = OutputHandler(metric_names=['accuracy','loss'],tag="validation")
tb_logger.attach(
    evaluator,
    metrics_logger,
    event_name=Events.EPOCH_COMPLETED
)

#train
train_loss_logger = OutputHandler(metric_names=["loss"],tag="train")
tb_logger.attach(
    trainer,
    train_loss_logger,
    event_name=Events.EPOCH_COMPLETED
)



trainer.run(train_dl,max_epochs=30)
