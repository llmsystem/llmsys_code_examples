This example is adopted from https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat, with some modifications.

### Request GPUs
```bash
srun --cpus-per-task=5 --gpus=4 --mem=256GB --partition=<partition> --time=10:34:56 --pty bash
```

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Finetuning LLaMA2-7b
If stage is not provided, the script will use DeepSpeed ZeRO stage 3 by default.
Using stage 0 would disable ZeRO. (You will probably see CUDA OOM errors!)
```bash
bash run_llama2_7b.sh <stage>
```
