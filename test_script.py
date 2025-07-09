from litgpt import LLM

# llm = LLM.load("/raid/longhorn/calvin/litgpt/out/pretrain/htx-sutd-tinyllama/final")
# llm
# llm.save_pretrained("/raid/longhorn/calvin/litgpt/out/pretrain/htx-sutd-tinyllama-convert")
# while True:
#     text = llm.generate(input("Input:"))
#     print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.

import torch
from transformers import AutoModel


state_dict = torch.load("/raid/longhorn/calvin/litgpt/out/pretrain/htx-sutd-tinyllama/final/lit_model.pth")
model = AutoModel.from_pretrained(
    "/raid/longhorn/calvin/litgpt/out/pretrain/htx-sutd-tinyllama-hf/", local_files_only=True, state_dict=state_dict
)


# python -m litgpt.finetune.full --config config_hub/finetune/htx-sutd/full_htxsutd_tinyllama.yaml --logger tensorboard