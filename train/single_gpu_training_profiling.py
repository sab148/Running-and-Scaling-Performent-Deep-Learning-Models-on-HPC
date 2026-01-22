import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset import LanguageModelingDataset, build_vocab
from model.transformerLM import TransformerLM, ModelArgs
from profiler.training_loop_profile import train_model_profile
from utils.profiler_utils import ProfilerSection, ExecutionTimer
from utils.logger_utils import LoggerUtils

def train_model(model, train_loader, vocab, optimizer, loss_func, device):
    """
        Train the model on the entire training dataset and return the global loss.
    """

    model.train()
    
    total_loss = 0

    for _, (src, tgt) in enumerate(train_loader):
        
        src, tgt = src.to(device), tgt.to(device)
        output = model(src)  # (seq_len, batch, vocab)
        
        loss = loss_func(output.view(-1, len(vocab)), tgt.t().reshape(-1))
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss

    result = total_loss / len(train_loader)

    return result


def test_model(model, dataloader, vocab, loss_func, device):
    """
        Evaluate the model on an evaluation set and return the global
        loss over the entire evaluation set.
    """
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            loss = loss_func(output.view(-1, len(vocab)), tgt.t().reshape(-1))
            total_loss += loss

    result = total_loss / len(dataloader)

    return result

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build vocab from training data
    vocab, stoi, itos = build_vocab('train')

    # Set up the datasets and dataloaders Shared across all splits
    train_dataset = LanguageModelingDataset('train', seq_len=32, stoi=stoi, vocab=vocab)
    val_dataset = LanguageModelingDataset('validation', seq_len=32, stoi=stoi, vocab=vocab)
    test_dataset = LanguageModelingDataset('test', seq_len=32, stoi=stoi, vocab=vocab)

    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, 
                            num_workers=4,
                            pin_memory=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            pin_memory=True)             


    # Set up the model and move it to the device
    model_args = ModelArgs(
        dim=128, 
        n_heads=4, 
        max_seq_length=2048, 
        vocab_size=len(vocab), 
        num_encoder_layers=2
    )
    model = TransformerLM(model_args)
    model = model.to(device)
    
    # Set up the loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    best_val_loss = float("inf")
    
    # Set up wandb or TensorBoard logging
    logger = LoggerUtils(args.logger, args.lr, args.epochs, args.batch_size)

    # Train the model
    for epoch in range(args.epochs):
        
        start_time = time.time()

        train_loss = train_model_profile(model, train_loader, vocab, optimizer, loss_func, device)
    
        train_epoch_time = time.time() - start_time

        val_loss = test_model(model, val_loader, vocab, loss_func, device)

        print(f'[{epoch+1}/{args.epochs}] Train loss: {train_loss:.5f}, Validation loss: {val_loss:.5f}') 
        print(f'[{epoch+1}/{args.epochs}] Epoch_Time (Training): {train_epoch_time:.5f}') 

        logger.log_metrics(train_loss, val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model, 'model_best.pt')

    
    test_loss = test_model(model, test_loader, vocab, loss_func, device)
    print('Final test loss:', test_loss.item()) 

    torch.save(model, 'model-final.pt')

   # Close wandb or the TensorBoard writer
    logger.close_logger()
        


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Single GPU Training')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size ')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=.002,
                        help='learning rate (default: .002)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--profile', action='store_true',
                        help='enable profiling')
    parser.add_argument('--logger', type=str, default='tensorboard',
                        choices=['tensorboard', 'wandb'],
                        help='logger to use (default: tensorboard)')
    args = parser.parse_args()

    if args.profile:
        torch.multiprocessing.set_start_method("spawn", force=True)
    torch.manual_seed(args.seed)

    main(args)
