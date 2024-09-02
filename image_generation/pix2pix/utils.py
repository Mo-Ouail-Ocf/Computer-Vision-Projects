from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

from ignite.engine import Engine,Events

from torchvision.utils import make_grid 

from ignite.contrib.handlers import tqdm_logger

from data import valid_dl

def attach_ignite(
        trainer:Engine,
        evaluator:Engine
):
    
    tb_logger = TensorboardLogger(log_dir ='./pix2pix_log')

    tqdm_train = tqdm_logger.ProgressBar().attach(trainer)
    

    def log_generated_images(engine, logger, event_name):

        global_step = trainer.state.epoch
        state = engine.state

        input_images = (state.output['input_images']+1)/2 *255
        generated_images = (state.output['generated_images']+1)/2 * 255
        target_images = (state.output['target_images']+1)/2*255
        
        input_grid = make_grid(input_images, padding=2)
        generated_grid = make_grid(generated_images, padding=2)
        target_grid = make_grid(target_images, padding=2)
        
        logger.writer.add_image(tag='Input Images', img_tensor=input_grid, global_step=global_step, dataformats='CHW')
        logger.writer.add_image(tag='Generated Images', img_tensor=generated_grid, global_step=global_step, dataformats='CHW')
        logger.writer.add_image(tag='Target Images', img_tensor=target_grid, global_step=global_step,dataformats='CHW')

    tb_logger.attach(evaluator, 
                 log_handler=log_generated_images, 
                 event_name=Events.EPOCH_COMPLETED)
    
    tb_logger.attach_output_handler(
        engine=trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag='train',
        output_transform=lambda x:x
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_valid(engine):
        evaluator.run(valid_dl)

    

    
    
