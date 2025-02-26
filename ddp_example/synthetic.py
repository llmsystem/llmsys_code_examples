import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def create_dataset(num_samples):
    # Here we create a synthetic dataset
    inputs = torch.randn(num_samples, 10)
    labels = torch.randn(num_samples, 5)
    return TensorDataset(inputs, labels)


def train_single_gpu(dataset, batch_size=1024, num_epochs=100):
    # The following code only train with one gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ToyModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    end_time = time.time()
    
    return end_time - start_time


def train_ddp(dataset, batch_size=1024, num_epochs=100):
    # We initialize process group with nccl for communication
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # We sample using DistributedSampler such that each GPU gets unique data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size // world_size, sampler=sampler)

    start_time = time.time()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # we will have different shuffling per epoch
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device_id), labels.to(device_id)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    end_time = time.time()
    
    dist.destroy_process_group()
    return end_time - start_time


if __name__ == "__main__":
    import sys

    NUM_SAMPLES = 100000
    BATCH_SIZE = 1024
    NUM_EPOCHS = 100

    dataset = create_dataset(NUM_SAMPLES)

    if "--ddp" in sys.argv:
        # Run DDP training
        total_time = train_ddp(dataset, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)
        print(f"DDP Training Time: {total_time:.4f} seconds")
    
    else:
        # Run Single-GPU training
        total_time = train_single_gpu(dataset, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)
        print(f"Single-GPU Training Time: {total_time:.4f} seconds")

