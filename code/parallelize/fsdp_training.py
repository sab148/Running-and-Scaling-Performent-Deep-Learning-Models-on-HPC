import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import LanguageModelingDataset, build_vocab
from transformerLM import TransformerLM, ModelArgs
# This file contains utility_functions for distributed training.
from distributed_utils import *

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
    # Return the global average loss.
    torch.distributed.all_reduce(result, torch.distributed.ReduceOp.AVG)

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
    # Return the global average loss.
    torch.distributed.all_reduce(result, torch.distributed.ReduceOp.AVG)

    return result

def main(args):

    # Initialize a communication group and return the right identifiers.
    local_rank, rank, device, world_size = setup()

    # Build vocab from training data
    vocab, stoi, itos = build_vocab('train')

    # Set up the datasets and dataloaders Shared across all splits
    train_dataset = LanguageModelingDataset('train', seq_len=32, stoi=stoi, vocab=vocab)
    val_dataset = LanguageModelingDataset('validation', seq_len=32, stoi=stoi, vocab=vocab)
    test_dataset = LanguageModelingDataset('test', seq_len=32, stoi=stoi, vocab=vocab)

    # DistributedSampler object for each set to ensure that each process gets a different subset of the data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, 
                                                                    shuffle=True, 
                                                                    seed=args.seed)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, 
                            sampler=train_sampler, # pass the sampler argument to the DataLoader
                            num_workers=int(os.getenv('SLURM_CPUS_PER_TASK')),
                            pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            sampler=val_sampler, # pass the sampler argument to the DataLoader
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            sampler=test_sampler, # pass the sampler argument to the DataLoader
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
    
    # Unlike DDP, we should apply fully_shard to both submodules and the root model.
    # Here, we apply fully_shard to each TransformerEncoder and TransformerDecoder block,
    # and then to the root model.
    fsdp_kwargs = {}
    for module in model.modules():
        if isinstance(module, (
                torch.nn.TransformerEncoder, 
                torch.nn.TransformerDecoder,)
            ):
            # Each TransformerEncoder and TransformerDecoder block is treated as a separate FSDP unit.
            torch.distributed.fsdp.fully_shard(module, **fsdp_kwargs)

    # Identifies all parameters not already wrapped and groups them into a shardable unit.
    torch.distributed.fsdp.fully_shard(model, **fsdp_kwargs)
    
    # Set up the loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    best_val_loss = float("inf")

    # Train the model
    for epoch in range(args.epochs):
        # Pass the current epoch to the sampler to ensure proper data shuffling in each epoch
        train_sampler.set_epoch(epoch)

        train_loss = train_model(model, train_loader, vocab, optimizer, loss_func, device)
        val_loss = test_model(model, val_loader, vocab, loss_func, device)

        # We use the utility function print0 to print messages only from rank 0.
        print0(f'[{epoch+1}/{args.epochs}] Train loss: {train_loss:.5f}, validation loss: {val_loss:.5f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save sharded model and optimizer
            save_sharded_model(model, optimizer, 'model_best')

    
    test_loss = test_model(model, test_loader, vocab, loss_func, device)
    ## TODO 11: Replace print by print0 to print messages once.
    print0('Final test loss:', test_loss.item()) 

    # Save sharded model and optimizer
    save_sharded_model(model, optimizer, 'model_final')

    # Destroy the process group to clean up resources
    destroy_process_group()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Single GPU Training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size ')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=.002,
                        help='learning rate (default: .002)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    main(args)
