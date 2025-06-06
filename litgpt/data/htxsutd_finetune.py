# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union, TypedDict, Literal

import torch
from torch.utils.data import DataLoader, random_split

from litgpt.data import DataModule, get_sft_collate_fn, SFTMultiTurnDataset
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer

TRAIN_KEY = "train_sft"
VAL_KEY = "test_sft" # should be val, but some dataset is test
PAD_ID = 128004 # LLAMA>3.1 pad id

class PreparedUltraChat(TypedDict):
    train_dataset: SFTMultiTurnDataset
    val_dataset: None
    test_dataset: None


class UltraChatMessage(TypedDict):
    content: str
    role: Literal['user', 'assistant']


class UltraChatRow(TypedDict):
    prompt: str
    prompt_id: str
    messages: List[UltraChatMessage]

@dataclass
class HTXSUTDFinetune(DataModule):
    """LIMA data module for supervised finetuning."""

    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    prompt_style: Union[str, PromptStyle] = "llama3"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    include_multiturn_conversations: bool = True
    """Whether to include multi-turn conversations in the dataset."""
    repo_id: str = "HuggingFaceH4/ultrachat_200k" # TODO: Change this when the dataset is up
    """The Hugging Face dataset repository ID from where to download the data."""
    access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))
    """The Hugging Face API token to use for authentication. Can also be set through the
    `HF_TOKEN` environment variable."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTMultiTurnDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTMultiTurnDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        # if self.access_token is None:
        #     raise ValueError(
        #         "HTXSUTDFinetune requires authentication, please set the `HF_TOKEN=your_token` environment"
        #         " variable or pass --access_token=your_token. You can find your token by visiting"
        #         " https://huggingface.co/settings/tokens"
        #     )
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        from datasets import load_dataset

        load_dataset(self.repo_id, token=self.access_token)

    def setup(self, stage: str = "") -> None:
        from datasets import load_dataset

        dataset = load_dataset(self.repo_id, token=self.access_token)

        train_data = format_dataset(dataset[TRAIN_KEY], self.include_multiturn_conversations)
        test_data = format_dataset(dataset[VAL_KEY], self.include_multiturn_conversations)
        
        train_data, test_data = list(train_data), list(test_data)

        self.train_dataset = SFTMultiTurnDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTMultiTurnDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=PAD_ID, pad_id=PAD_ID),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=PAD_ID, pad_id=PAD_ID),
        )


def format_dataset(
    dataset: List[UltraChatRow], include_multi_turn_conversations: bool
):
    formatted = []

    for entry in dataset:
        formatted_convo = []
        convo = entry['messages']

        # Each conversation is a flat list of user-assistant pairs.
        # So we iterate in 2-step manner
        for i in range(0, len(convo) - 1, 2):
            if convo[i]['role'] != 'user':
                print(
                    f'WARN: UltraChat row with prompt_id[{entry["prompt_id"]}] is corrupted. Expected role to be `user`, but is `{convo[i]["role"]}` instead.'
                )
            if convo[i + 1]['role'] != 'assistant':
                print(
                    f'WARN: UltraChat row with prompt_id[{entry["prompt_id"]}] is corrupted. Expected role to be `assistant`, but is `{convo[i+1]["role"]}` instead.'
                )

            formatted_sft_dict = {
                'instruction': convo[i]['content'],
                'input': '',
                'output': convo[i + 1]['content'],
            }

            formatted_convo.append(formatted_sft_dict)

            # If don't want to include multi turn, break after first
            # turn is appended: - no point including latter turns as
            # they become orphaned discussions without starting context
            if not include_multi_turn_conversations:
                break

        formatted.append(formatted_convo)

    return formatted


foo = HTXSUTDFinetune()
from litgpt import Tokenizer

tokenizer = Tokenizer('/raid/longhorn/huangchen/models/Llama-3.1-8B-Instruct')

foo.connect(batch_size=4, tokenizer=tokenizer)
foo.setup()
data = foo.train_dataloader()

for d in data:
    print(d)
    input()

foo.setup() 

