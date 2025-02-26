import os
from io import open
import torch
from torch.utils.data import Dataset, DataLoader

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

class TextDataset(Dataset):
    """Formats corpus into sequences for language modeling"""
    def __init__(self, data, seq_len, device, batch_size = 128):
        self.device = device
        self.seq_len = seq_len
        data = data.narrow(0, 0, data.size(0) // batch_size * batch_size)
        self.data = data.view(batch_size, -1).t().contiguous().to(device)
        self.device = device

        self.indices = list(range(0, self.data.size(0)-self.seq_len, self.seq_len))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        inputs = self.data[start_idx: start_idx + self.seq_len]
        targets = self.data[start_idx + 1: start_idx + self.seq_len + 1].view(-1)
        return inputs.to(self.device), targets.to(self.device)

def get_dataloader(dataset, seq_len, batch_size=1, sampler=None):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
    )
    return loader

