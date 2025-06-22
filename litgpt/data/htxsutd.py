# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import os
from torch.utils.data import DataLoader

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer


import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

# PAD_ID = 128004
PAD_ID = 2


class ParquetDataset(Dataset):
    def __init__(self, parquet_files, seq_length, tokenizer: Tokenizer):
        # Concatenate all parquet files into a single DataFrame
        self.df = pd.concat([pd.read_parquet(file) for file in parquet_files])
        self.seq_length = seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.df.iloc[idx]["problem"], eos=True)

        # tokens = self.df.iloc[idx]['tokens']  # adjust column name if needed
        # tokens = torch.tensor(tokens, dtype=torch.long)

        # Ensure tokens are at least seq_length + 1 (pad or truncate)
        if len(tokens) < self.seq_length + 1:
            pad = torch.full((self.seq_length + 1 - len(tokens),), fill_value=PAD_ID, dtype=torch.long)
            tokens = torch.cat([tokens, pad], dim=0)
        else:
            tokens = tokens[: self.seq_length + 1]

        # print(tokens)

        return tokens


@dataclass
class HTXSUTD(DataModule):
    """The HTX SUTD data module is composed of data filtered from FineWeb and information crawled from HomeTeam websites.

    Provides training and validation streaming dataloaders that return batches of tokens.
    """

    # data_path: Union[str, Path] = Path("data/")
    data_path: Union[str, Path] = Path("../../huangchen/Parquet_files")
    """The path to the data directory"""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 8
    """How many DataLoader processes to use for loading."""
    use_starcoder: bool = True
    """Toggle for using Starcoder data."""

    batch_size: int = field(init=False, repr=False, default=1)
    # seq_length: int = field(init=False, repr=False, default=131072)
    seq_length: int = field(init=False, repr=False, default=4096)

    # def __init__(self, data_path = Path("data/")):
    #     self.data_path = data_path

    def __post_init__(self):
        super().__init__()
        # Could be a remote path (s3://) or a local path
        self.train = str(self.data_path).rstrip("/") + "/train"
        self.val = str(self.data_path).rstrip("/") + "/val"
        self.required_paths = [self.train, self.val]

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        for path in self.required_paths:
            if not path.startswith("s3://") and not Path(path).is_dir():
                raise FileNotFoundError(
                    "The data path for HTXSUTD is expected to be the directory containing these subdirectories:"
                    f" `train`, `val`. The directory {path} does not exist."
                    " Set it via `--data.data_path=...`"
                )

    def train_dataloader(self) -> DataLoader:
        # Create list of Parquet files (adjust as needed)
        parquet_files = [
            os.path.join(self.train, f) for f in os.listdir(self.train) if f.endswith("tokenized.parquet")
        ]  # Replace with actual file paths
        # Create dataset
        dataset = ParquetDataset(parquet_files, seq_length=self.seq_length, tokenizer=self.tokenizer)

        # Create DataLoader
        train_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        # Create list of Parquet files (adjust as needed)
        parquet_files = [
            os.path.join(self.val, f) for f in os.listdir(self.val) if f.endswith("tokenized.parquet")
        ]  # Replace with actual file paths

        # Create dataset
        dataset = ParquetDataset(parquet_files, seq_length=self.seq_length, tokenizer=self.tokenizer)

        # Create DataLoader
        val_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

        return val_dataloader
