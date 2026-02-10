import argparse
import time

import torch
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader

from dataset.dataset import LanguageModelingDataset, build_vocab
from model.transformerLM import TransformerLM, ModelArgs
from utils.logger_utils import LoggerUtils
from utils.distributed_utils import *
from utils.tp_utils import apply_tensor_parallelism, _replicate_tensor


def train_model(model, train_loader, optimizer, loss_func, device_mesh):
    """
        Train the model on the entire training dataset and return the global loss.
    """

    model.train()
    
    total_loss = 0

    for _, (src, tgt) in enumerate(train_loader):

        src = _replicate_tensor(src, device_mesh)
        tgt = _replicate_tensor(tgt, device_mesh)

        output = model(src)  # (seq_len, batch, vocab)
        
        loss = loss_func(output.view(-1, output.size(-1)), tgt.t().reshape(-1))
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss

    result = total_loss / len(train_loader)

    return result.item()


def test_model(model, dataloader, loss_func, device_mesh):
    """
        Train the model on the entire training dataset and return the global loss.
    """

    model.eval()
    
    total_loss = 0

    with torch.no_grad():
        for _, (src, tgt) in enumerate(dataloader):

            src = _replicate_tensor(src, device_mesh)
            tgt = _replicate_tensor(tgt, device_mesh)

            output = model(src)  # (seq_len, batch, vocab)
            
            # output = output.transpose(0, 1).contiguous()  # (seq, batch, vocab) -> (batch, seq, vocab)
            
            # loss = loss_func(
            #     output.view(-1, output.size(-1)), 
            #     tgt.view(-1)
            # )
            loss = loss_func(output.view(-1, output.size(-1)), tgt.t().reshape(-1))
        
            total_loss += loss

    result = total_loss / len(dataloader)

    return result.item()


def main(args):
    # Initialize distributed
    local_rank, rank, device, world_size = setup()
    
    # Create device mesh for tensor parallelism
    device_mesh = init_device_mesh("cuda", (world_size,))
    
    # Build vocab from training data
    vocab, stoi, itos = build_vocab('train')

    # Set up the datasets and dataloaders Shared across all splits
    train_dataset = LanguageModelingDataset('train', seq_len=32, stoi=stoi, vocab=vocab)
    val_dataset = LanguageModelingDataset('validation', seq_len=32, stoi=stoi, vocab=vocab)
    test_dataset = LanguageModelingDataset('test', seq_len=32, stoi=stoi, vocab=vocab)

    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=4,
                            pin_memory=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            pin_memory=True)             

    # Create model
    model_args = ModelArgs(
        dim=128, 
        n_heads=4, 
        max_seq_length=2048, 
        vocab_size=len(vocab), 
        num_encoder_layers=2
    )
    model = TransformerLM(model_args)
    model = model.to(device)

    model = apply_tensor_parallelism(model, device_mesh, rank)
    
    # Create optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float("inf")

    # Set up TensorBoard logging
    logger = LoggerUtils(args.logger, args.lr, args.epochs, args.batch_size)

    # Train the model
    for epoch in range(args.epochs):
        # Generate random data (in practice, use real data)
        start_time = time.time()
        train_loss = train_model(model, train_loader, optimizer, loss_func, device_mesh)
        train_epoch_time = time.time() - start_time
        val_loss = test_model(model, val_loader, loss_func, device_mesh)

        # We use the utility function print0 to print messages only from rank 0.
        print0(f'[{epoch+1}/{args.epochs}] Train loss: {train_loss:.5f}, Validation loss: {val_loss:.5f}') 
        print0(f'[{epoch+1}/{args.epochs}] Epoch_Time (Training): {train_epoch_time:.5f}') 

        # Log metrics to the logger
        logger.log_metrics(train_loss, val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            ## TODO 18: Replace save0 method by either save_full_model or save_sharded_model to save the full model state or the sharded model state respectively.
            # We allow only rank=0 to save the model
            save0(model, 'model_best.pt')


    test_loss = test_model(model, test_loader, loss_func, device_mesh)
    # We use the utility function print0 to print messages only from rank 0.
    print0('Final test loss:', test_loss.item()) 

    ## TODO 18: Replace save0 method by either save_full_model or save_sharded_model to save the full model state or the sharded model state respectively.
    # We allow only rank=0 to save the model
    save0(model, 'model_final.pt')

    # Close the logger
    logger.close_logger()

    # Cleanup
    destroy_process_group()


if __name__ == "__main__":
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
