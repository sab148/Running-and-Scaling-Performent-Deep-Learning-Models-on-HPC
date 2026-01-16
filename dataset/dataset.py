import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Function to build vocab from training split
def build_vocab(split='train'):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    tokens = ' '.join(dataset['text']).lower().split()
    vocab = sorted(set(tokens))
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return vocab, stoi, itos

# Dataset for language modeling
class LanguageModelingDataset(Dataset):
    def __init__(self, split, seq_len=256, stoi=None, vocab=None):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        tokens = ' '.join(dataset['text']).lower().split()

        if stoi is None or vocab is None:
            raise ValueError("Must pass shared stoi and vocab from training split.")

        self.vocab = vocab
        self.stoi = stoi
        self.data = torch.tensor([self.stoi[t] for t in tokens if t in self.stoi], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + 1 + self.seq_len]
        return x, y
