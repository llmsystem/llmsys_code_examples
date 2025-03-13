import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import matplotlib.pyplot as plt

import data
import model
from data import TextDataset, get_dataloader

def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()

def cleanup_ddp():
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')
    parser.add_argument('--ddp', action='store_true', help='run with ddp', default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    return parser.parse_args()

def plot_training_metrics(train_losses, val_losses, ppl_values, save_path='training_metrics.png', ddp=False):
    if ddp:
        save_path = save_path.replace('.png', '_ddp.png')

    epochs = range(1, len(train_losses) + 1)
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    axs[0].plot(epochs, train_losses, marker='o', linestyle='-', color='blue', label='Training Loss')
    axs[1].plot(epochs, val_losses, marker='o', linestyle='-', color='red', label='Validation Loss')
    axs[2].plot(epochs, ppl_values, marker='o', linestyle='-', color='green', label='Perplexity')
    for ax, title in zip(axs, ['Training Loss', 'Validation Loss', 'Perplexity']):
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")

class Trainer:
    def __init__(self, args, model, train_loader, val_loader, test_loader, ntokens, device):
        self.args = args
        self.device = device
        self.ntokens = ntokens
        self.model = model.to(device)
        self.lr = 0.05
        if args.ddp:
            self.model = DDP(self.model, device_ids=[device.index])
            self.lr *= dist.get_world_size() #scale learning rate to compensate gradient averaging
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.best_val_loss = None

    def train_one_epoch(self, epoch):
        if self.args.ddp:
            self.train_loader.sampler.set_epoch(epoch)

        self.model.train()
        total_loss = 0.
        start_time = time.time()
        for data, targets in self.train_loader:
            self.model.zero_grad()
            data = data.squeeze(0).clone()
            targets = targets.squeeze(0).clone()
            output = self.model(data)
            targets = targets.view(-1)
            output = output.view(-1, self.ntokens)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()
            total_loss = total_loss + loss.item()

        if self.args.ddp:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)

            return (total_loss_tensor / dist.get_world_size() / len(self.train_loader)).item()
        else:
            return total_loss / len(self.train_loader)

    def evaluate_model(self, loader):
        self.model.eval()
        total_loss = 0.
        with torch.no_grad():
            for data, targets in loader:
                data = data.squeeze(0)
                targets = targets.squeeze(0)
                output = self.model(data)
                output = output.view(-1, self.ntokens)
                total_loss = total_loss + self.criterion(output, targets).item()

        if self.args.ddp:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)

            return (total_loss_tensor / dist.get_world_size() / len(loader)).item()
        else:
            return total_loss / len(loader)

    def train_model(self, num_epochs=5):
        train_losses, val_losses, ppl_values = [], [], []
        try:
            for epoch in range(1, num_epochs + 1):
                epoch_start_time = time.time()
                train_loss = self.train_one_epoch(epoch)
                val_loss = self.evaluate_model(self.val_loader)
                ppl = math.exp(val_loss)
                
                if not self.args.ddp or (self.args.ddp and self.device.index == 0):
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    ppl_values.append(ppl)
                    print(f'| Epoch {epoch} | Time: {time.time() - epoch_start_time:.2f}s | Train Loss {train_loss:.2f} | Valid Loss {val_loss:.2f} | Perplexity {ppl:.2f}')
                    
                    if self.best_val_loss is None or val_loss < self.best_val_loss:
                        torch.save(self.model, 'model.pt')
                        self.best_val_loss = val_loss
                    else:
                        self.lr /= 4.0
        except KeyboardInterrupt:
            print('Exiting from training early')
        
        self.test_model()
        if not self.args.ddp or (self.args.ddp and self.device.index == 0):
            plot_training_metrics(train_losses, val_losses, ppl_values, ddp=self.args.ddp)

    def test_model(self):
        test_loss = self.evaluate_model(self.test_loader)
        if not self.args.ddp or (self.args.ddp and self.device.index == 0):
            print(f'| End of Training | Test Loss {test_loss:.2f} | Test Perplexity {math.exp(test_loss):.2f}')


def main():
    args = parse_args()
    
    if args.ddp:
        rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    corpus = data.Corpus('./data/wikitext-2')
    ntokens = len(corpus.dictionary)
    seq_len = 40
    
    corpus.train = TextDataset(corpus.train, seq_len, device, args.batch_size)
    corpus.valid = TextDataset(corpus.valid, seq_len, device, args.batch_size)
    corpus.test = TextDataset(corpus.test, seq_len, device, args.batch_size)

    sampler = DistributedSampler(corpus.train, world_size, rank, shuffle=False) if args.ddp else None
    train_loader = get_dataloader(corpus.train, seq_len=40, sampler=sampler)
    val_loader = get_dataloader(corpus.valid, seq_len=40)
    test_loader = get_dataloader(corpus.test, seq_len=40) 

    model_inst = model.TransformerModel(ntokens, 256, 8, 256, 4, 0.2)
    trainer = Trainer(args, model_inst, train_loader, val_loader, test_loader, ntokens, device)
    trainer.train_model()

    if args.ddp:
        cleanup_ddp()

if __name__ == "__main__":
    main()
