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

# TODO: Test code
# def load_first_n_rows(parquet_path, n_rows=10000):
#     import pyarrow.parquet as pq

#     pf = pq.ParquetFile(parquet_path)

#     batches = []
#     total_rows = 0

#     for i in range(pf.num_row_groups):
#         rg_table = pf.read_row_group(i)
#         rg_df = rg_table.to_pandas()
#         rows_needed = n_rows - total_rows

#         if len(rg_df) > rows_needed:
#             batches.append(rg_df.iloc[:rows_needed])
#             break
#         else:
#             batches.append(rg_df)
#             total_rows += len(rg_df)

#         if total_rows >= n_rows:
#             break

#     return pd.concat(batches, ignore_index=True)


def load_parquet(folder: Path, seed: int, shuffle=True):
    # Get all *.parquet files in the folder
    parquet_files = list(folder.glob("*.parquet"))

    if len(parquet_files) == 0:
        raise Exception(f"No files found in [{folder}]")

    # Load all files into a single DataFrame
    df = pd.concat([pd.read_parquet(file, engine="pyarrow") for file in parquet_files], ignore_index=True)

    # TODO: Test code
    # df = pd.concat([load_first_n_rows(file) for file in parquet_files], ignore_index=True)

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


class ParquetDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, seq_length, tokenizer: Tokenizer):
        # Concatenate all parquet files into a single DataFrame
        self.df = dataframe
        self.seq_length = seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Note: bos is automatically handled based on tokenizer. See tokenizer.py -> #check_if_bos_token_used
        #       It is added for all LlamaTokenizer based on tokenizer_config.json
        tokens = self.tokenizer.encode(self.df.iloc[idx]["text"], eos=True)

        # Ensure tokens are at least seq_length + 1 (pad or truncate)
        if len(tokens) < self.seq_length + 1:
            pad = torch.full((self.seq_length + 1 - len(tokens),), fill_value=self.tokenizer.pad_id, dtype=torch.long)
            tokens = torch.cat([tokens, pad], dim=0)
        else:
            tokens = tokens[: self.seq_length + 1]

        return tokens


@dataclass
class HTXSUTD(DataModule):
    """The HTX SUTD data module is composed of data filtered from FineWeb and information crawled from HomeTeam websites.

    Provides training and validation streaming dataloaders that return batches of tokens.
    """

    dataset_path: Path = Path("dataset/HTXSUTD-pretrain")
    """The path to the data directory"""
    num_workers: int = 8
    """How many DataLoader processes to use for loading."""

    batch_size: int = field(init=False, repr=False, default=1)

    # doesn't matter will just follow model's max seq length
    max_seq_length: int = field(init=False, repr=False, default=2048)

    tokenizer: Optional[Tokenizer] = None

    shuffle: bool = True

    def __post_init__(self):
        super().__init__()
        # Could be a remote path (s3://) or a local path
        self.train_path = Path(str(self.dataset_path).rstrip("/") + "/train")
        self.val_path = Path(str(self.dataset_path).rstrip("/") + "/validation")
        self.required_paths = [self.train_path, self.val_path]

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

        if self.tokenizer is None:
            raise ValueError("Tokenizer is needed.")

        if self.tokenizer.pad_id is None:
            raise ValueError("Tokenizer must have `pad_id`.")

    def prepare_data(self) -> None:
        for path in self.required_paths:
            if not str(path).startswith("s3://") and not path.is_dir():
                raise FileNotFoundError(
                    "The data path for HTXSUTD is expected to be the directory containing these subdirectories:"
                    f" `train`, `val`. The directory {str(path)} does not exist."
                    " Set it via `--data.dataset_path=...`"
                )

    def train_dataloader(self) -> DataLoader:
        dataframe = load_parquet(self.train_path, seed=self.seed, shuffle=self.shuffle)
        dataset = ParquetDataset(dataframe, seq_length=self.max_seq_length, tokenizer=self.tokenizer)

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
        # TODO: Test code
        # dataframe = load_parquet_val_temp(self.val_path, seed=self.seed, shuffle=self.shuffle)
        dataframe = load_parquet(self.val_path, seed=self.seed, shuffle=self.shuffle)
        dataset = ParquetDataset(dataframe, seq_length=self.max_seq_length, tokenizer=self.tokenizer)

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
