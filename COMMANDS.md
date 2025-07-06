## First finetune test run

```bash
WANDB_PROJECT=HTX_SUTD WAND_NAME=8b-test-run python -m litgpt.finetune.full --config config_hub/finetune/llama-3.1-8b/full_htxsutd.yaml
```

## Pretrain test run

```bash
WANDB_PROJECT=HTX_SUTD WANDB_NAME=1b-pretrain-run1 python -m litgpt.pretrain --config config_hub/pretrain/htxsutd-tinyllama.yaml --logger_name wandb
```