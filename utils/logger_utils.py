import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

class LoggerUtils:
    def __init__(self, logger, lr, epochs, batch_size):
        self.logger = logger
        self.writer = None
        self.wandb = None
        
        if (
            (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)
            or (not torch.distributed.is_initialized())
        ):
            if logger == 'tensorboard':    
                self.writer = SummaryWriter("tensorboard_logs")
            elif logger == 'wandb':            
                # Initialize Weights & Biases
                wandb.init(project="wandb_distributed_training",  # Changed from self.wandb.init
                           name=f"single_gpu_training_run",
                           mode="offline",
                           reinit=True)
                wandb.config.update({"learning_rate": lr,
                                     "epochs": epochs,
                                     "batch_size": batch_size})
                self.wandb = wandb  # Store the wandb module for later use
    
    def log_metrics(self, train_loss, val_loss, epoch):
        if (
            (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)
            or (not torch.distributed.is_initialized())
        ):
            if self.logger == 'tensorboard':
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
            elif self.logger == 'wandb':
                wandb.log({"Loss/Train": train_loss, "Loss/Validation": val_loss, "Epoch": epoch})
    
    def close_logger(self):
        if (
            (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)
            or (not torch.distributed.is_initialized())
        ):  
            if self.logger == 'tensorboard':
                self.writer.close()
            elif self.logger == 'wandb':
                wandb.finish()