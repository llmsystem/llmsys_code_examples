# Language Model Training Example

This example is adapted from [PyTorch Word Language Model](https://github.com/pytorch/examples/tree/main/word_language_model) and demonstrates how to train a language model using PyTorch, both on a single GPU and using Distributed Data Parallel (DDP) for multiple GPUs.

## Instructions

### Running Training on a Single GPU
To train the model on a single GPU, run the following command:
```
torchrun --nproc_per_node=1 main.py
```

This will start training with the default settings. You should see output similar to:
```
| Epoch 1 | Time: 13.50s | Train Loss 7.85 | Valid Loss 7.21 | Perplexity 1355.19
| Epoch 2 | Time: 13.32s | Train Loss 7.30 | Valid Loss 6.93 | Perplexity 1026.65
| Epoch 3 | Time: 13.36s | Train Loss 7.09 | Valid Loss 6.79 | Perplexity 885.36
| Epoch 4 | Time: 13.48s | Train Loss 6.97 | Valid Loss 6.69 | Perplexity 805.83
| Epoch 5 | Time: 13.52s | Train Loss 6.89 | Valid Loss 6.62 | Perplexity 746.42
| End of Training | Test Loss 6.53 | Test Perplexity 687.53
```

### Running Distributed Data Parallel (DDP) Training on Multiple GPUs
Replace <num_of_gpus> with the number of GPUs you wish to use. For example, to use 2 GPUs:
```
torchrun --nproc_per_node=2 main.py --ddp
```

This will start the training process with DDP enabled, and you should see output similar to:
```
| Epoch 1 | Time: 7.48s | Train Loss 8.14 | Valid Loss 7.47 | Perplexity 1757.71
| Epoch 2 | Time: 7.28s | Train Loss 7.56 | Valid Loss 7.21 | Perplexity 1353.03
| Epoch 3 | Time: 7.29s | Train Loss 7.36 | Valid Loss 7.05 | Perplexity 1154.24
| Epoch 4 | Time: 7.31s | Train Loss 7.23 | Valid Loss 6.94 | Perplexity 1027.78
| Epoch 5 | Time: 7.32s | Train Loss 7.13 | Valid Loss 6.85 | Perplexity 945.97
| End of Training | Test Loss 6.77 | Test Perplexity 873.09
```

You will see the train, validation loss and perplexity plots (training\_metrics*.png).

### Debugging DDP
If the ddp initialization is stucked, it is likely that your GPUs are not connected via NVLink, please use `export NCCL_P2P_DISABLE=1`.
