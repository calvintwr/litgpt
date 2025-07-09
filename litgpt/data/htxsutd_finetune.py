# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union, TypedDict, Literal
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split, Dataset

from litgpt.data import DataModule, get_sft_collate_fn
from litgpt.data.sft_multi_turn_base import SFTMultiTurnDataset
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer
import pandas as pd

TRAIN_KEY = "train"
VAL_KEY = "validation"  # should be val, but some dataset is test


class PreparedUltraChat(TypedDict):
    train_dataset: SFTMultiTurnDataset
    val_dataset: None
    test_dataset: None


class UltraChatMessage(TypedDict):
    content: str
    role: Literal["user", "assistant"]


class UltraChatRow(TypedDict):
    prompt: str
    unique_id: str
    conversation: List[UltraChatMessage]
    actual_num_turns: int


def load_parquet(folder: Path, seed=42, shuffle=True):
    # Get all *.parquet files in the folder (or "*.tokenized.parquet" if needed)
    parquet_files = list(folder.glob("*.parquet"))  # or use "*.tokenized.parquet"

    if len(parquet_files) == 0:
        raise Exception(f"No files found in [{folder}]")

    # Load all files into a single DataFrame
    df = pd.concat([pd.read_parquet(file, engine="pyarrow") for file in parquet_files], ignore_index=True)
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


@dataclass
class HTXSUTDFinetune(DataModule):
    """LIMA data module for supervised finetuning."""

    mask_prompt: bool = True
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
    # repo_id: str = "HuggingFaceH4/ultrachat_200k"  # TODO: Change this when the dataset is up
    dataset_path: Path = Path("dataset/HTXSUTD-finetune")
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

        if self.tokenizer is None:
            raise ValueError("Tokenizer is needed.")

        if self.tokenizer.pad_id is None:
            raise ValueError("Tokenizer must have `pad_id`.")

    def prepare_data(self) -> None:
        # TODO: We are loading from local for now. Write this up if dataset is published on HF
        # from datasets import load_dataset

        # load_dataset(self.repo_id, token=self.access_token)
        return True

    def setup(self, stage: str = "") -> None:
        # Create dataset
        train_data = load_parquet(self.dataset_path / TRAIN_KEY, seed=self.seed)
        train_data = format_dataset(train_data, include_multi_turn_conversations=True)

        test_data = load_parquet(self.dataset_path / VAL_KEY, seed=self.seed)
        test_data = format_dataset(test_data, include_multi_turn_conversations=True)

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
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index,
                pad_id=self.tokenizer.pad_id,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index,
                pad_id=self.tokenizer.pad_id,
            ),
        )


def format_dataset(df: pd.DataFrame, include_multi_turn_conversations: bool):
    formatted = []

    for _, entry in df.iterrows():
        formatted_convo = []
        convo = entry["conversation"]

        # Each conversation is a flat list of user-assistant pairs.
        # So we iterate in 2-step manner
        for i in range(0, len(convo) - 1, 2):
            if convo[i]["role"] != "user":
                print(
                    f"WARN: UltraChat row with unique_id[{entry['unique_id']}] is corrupted. Expected role to be `user`, but is `{convo[i]['role']}` instead."
                )
            if convo[i + 1]["role"] != "assistant":
                print(
                    f"WARN: UltraChat row with unique_id[{entry['unique_id']}] is corrupted. Expected role to be `assistant`, but is `{convo[i + 1]['role']}` instead."
                )

            formatted_sft_dict = {
                "instruction": convo[i]["content"],
                "input": "",
                "output": convo[i + 1]["content"],
            }

            formatted_convo.append(formatted_sft_dict)

            # If don't want to include multi turn, break after first
            # turn is appended: - no point including latter turns as
            # they become orphaned discussions without starting context
            if not include_multi_turn_conversations:
                break

        formatted.append(formatted_convo)

    return formatted
