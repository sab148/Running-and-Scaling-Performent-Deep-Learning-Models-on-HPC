import os
import argparse

import torch
from torch import nn
from torch.nn import functional as F    
from torch.utils.data import DataLoader
import lightning as L

from model.transformerLM import LitTransformerLM, ModelArgs
from dataset.dataset import LanguageModelingDataset, build_vocab

def main(args):
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
                            pin_memory=True,
                            num_workers=4
                            )
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            pin_memory=True,
                            num_workers=4
                            )             


    model_args = ModelArgs(
        dim=128, 
        n_heads=4, 
        max_seq_length=2048, 
        vocab_size=len(vocab), 
        num_encoder_layers=2
    )
    
    model = LitTransformerLM(model_args, lr=args.lr)


    trainer = L.Trainer(
        devices=os.environ.get("SLURM_NTASKS_PER_NODE", "auto"),
        accelerator="gpu",
        max_epochs=args.epochs,
        num_nodes=int(os.getenv("SLURM_NNODES", "1")),  # Number of nodes from SLURM
    )


    trainer.fit(model, train_loader, val_loader)
    test_result = trainer.test(model, test_loader)
    print(test_result)



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
    args = parser.parse_args()


    main(args)
