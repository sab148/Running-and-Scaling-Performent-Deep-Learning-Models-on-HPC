import wandb
from torch.utils.tensorboard import SummaryWriter

class LoggerUtils:
    def __init__(self, logger, lr, epochs, batch_size):
        self.logger = None
        if logger == 'tensorboard':    
            self.writer = SummaryWriter("tensorboard_logs")
        elif logger == 'wandb':            
            # Initialize Weights & Biases
            self.wandb.init(project="wandb_distributed_training",
                        name=f"single_gpu_training_run",
                        reinit=True)
            self.wandb.config.update({"learning_rate": lr,
                                "epochs": epochs,
                                "batch_size": batch_size})


    def log_metrics(self, train_loss, val_loss, epoch):
        if self.logger == 'tensorboard':
            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
        elif self.logger == 'wandb':
            # Log metrics to Weights & Biases
            self.wandb.log({"Loss/Train": train_loss, "Loss/Validation": val_loss, "Epoch": epoch})


    def close_logger(self):
        if self.logger == 'tensorboard':
            self.writer.close()
        elif self.logger == 'wandb':
            self.wandb.finish