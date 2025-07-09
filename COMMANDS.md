## First finetune test run

```bash
WANDB_PROJECT=HTX_SUTD WANDB_NAME=8b-test-run python -m litgpt.finetune.full --config config_hub/finetune/llama-3.1-8b/full_htxsutd.yaml
```

## Pretrain runs

```bash
WANDB_PROJECT=HTX_SUTD WANDB_NAME=1b-pretrain-run1 python -m litgpt.pretrain --config config_hub/pretrain/htxsutd-tinyllama.yaml

WANDB_PROJECT=HTX_SUTD WANDB_NAME=3b-pretrain-run1 python -m litgpt.pretrain --config config_hub/pretrain/htxsutd-llama3b.yaml
```

### Conversion to checkpoint for finetuning:
```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true python -m litgpt.scripts.convert_pretrained_checkpoint /raid/longhorn/calvin/litgpt/out/pretrain/htx-sutd-tinyllama/final /raid/longhorn/calvin/litgpt/out/pretrain/htx-sutd-tinyllama-converted
```
Note: We need to use `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true` otherwise it will error out with `WeightsUnpickler error: Unsupported global: GLOBAL torch.utils.data.dataloader.DataLoader was not an allowed global by default.`. Long story short, DataLoader (along with a few other stuff) are serialized into the pretrain checkpoint. The error results because Torch forces loading of weights only and if it sees other objects, it deems the checkpoint unsafe to be loaded. We turn off this safety feature to convert.

## Finetune:

```bash
WANDB_PROJECT=HTX_SUTD WANDB_NAME=1b-finetune-run1  python -m litgpt.finetune.full --config config_hub/finetune/htx-sutd/full-htxsutd-tinyllama.yaml
```