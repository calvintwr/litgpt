# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Union, Literal
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, ShardingStrategy
import torch

_SHARDING_STRATEGY = Union[ShardingStrategy, Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]]


@dataclass
class TrainArgs:
    """Training-related arguments"""

    save_interval: Optional[int] = 1000
    """Number of optimizer steps between saving checkpoints"""
    log_interval: int = 1
    """Number of iterations between logging calls"""
    global_batch_size: int = 64
    """Number of samples between optimizer steps across data-parallel ranks"""
    micro_batch_size: int = 4
    """Number of samples per data-parallel rank"""
    lr_warmup_steps: Optional[int] = 100
    """Number of iterations with learning rate warmup active"""
    lr_warmup_fraction: Optional[float] = None
    """The fraction of an epoch to use for learning rate warmup"""
    epochs: Optional[int] = None
    """Number of epochs to train on"""
    # TODO: `pretrain` is the only script using `max_tokens` explicitly. replace it with epoch_size*epochs?
    max_tokens: Optional[int] = None
    """Total number of tokens to train on"""
    max_steps: Optional[int] = None
    """Limits the number of optimizer steps to run"""
    max_seq_length: Optional[int] = None
    """Limits the length of samples"""
    tie_embeddings: Optional[bool] = None
    """Whether to tie the embedding weights with the language modeling head weights"""

    # Optimization args
    max_norm: Optional[float] = None
    min_lr: float = 6e-5

    def __post_init__(self) -> None:
        if self.lr_warmup_fraction and self.lr_warmup_steps:
            raise ValueError(
                "Can't provide both `--train.lr_warmup_fraction` and `--train.lr_warmup_steps`. Choose one."
            )
        if self.lr_warmup_fraction and not (0 <= self.lr_warmup_fraction <= 1):
            raise ValueError("`--train.lr_warmup_fraction` must be between 0 and 1.")

        if self.lr_warmup_steps and self.max_steps and (self.lr_warmup_steps >= self.max_steps):
            warnings.warn(
                "`--train.lr_warmup_steps` should be less than `--train.max_steps`."
                f" Got {self.lr_warmup_steps} lr_warmup_steps and {self.max_steps} max_steps.",
                UserWarning,
            )

    def gradient_accumulation_iters(self, devices: int, num_nodes: int = 1) -> int:
        """Number of iterations between gradient synchronizations"""
        gradient_accumulation_iters = self.batch_size(devices, num_nodes) // self.micro_batch_size
        assert gradient_accumulation_iters > 0
        return gradient_accumulation_iters

    def batch_size(self, devices: int, num_nodes: int = 1) -> int:
        """Number of samples between optimizer steps per data-parallel rank"""
        batch_size = self.global_batch_size // (devices * num_nodes)
        assert batch_size > 0
        return batch_size

    def warmup_iters(self, devices: int, num_nodes: int, max_iters: int, train_dataloader) -> int:
        """Number of iterations to warm up the learning rate."""
        if self.lr_warmup_fraction:
            return min(max_iters, math.ceil(self.lr_warmup_fraction * len(train_dataloader)))
        if self.lr_warmup_steps:
            return min(max_iters, self.lr_warmup_steps * self.gradient_accumulation_iters(devices, num_nodes))
        return 0


@dataclass
class EvalArgs:
    """Evaluation-related arguments"""

    interval: int = 600
    """Number of optimizer steps between evaluation calls"""
    max_new_tokens: Optional[int] = None
    """Number of tokens to generate"""
    max_iters: int = 100
    """Number of iterations"""
    initial_validation: bool = False
    """Whether to evaluate on the validation set at the beginning of the training"""
    final_validation: bool = True
    """Whether to evaluate on the validation set at the end of the training"""
    evaluate_example: Union[str, int] = "first"
    """How to pick an example instruction to evaluate periodically during training.
       Can be "first", "random", or an integer index to pick a specific example."""


@dataclass
class LogArgs:
    """Logging-related arguments"""

    project: Optional[str] = None
    """Project name"""
    run: Optional[str] = None
    """Run name"""
    group: Optional[str] = None
    """Group name"""


@dataclass
class FSDPArgs:
    """FSDP arguments"""

    sharding_strategy: _SHARDING_STRATEGY = ShardingStrategy.SHARD_GRAD_OP
    """Sharding strategy. SHARD_GRAD_OP is the fastest but consumes 1x model parameter size more vram."""

    activation_checkpointing: bool = False
    """Turn on to save memory by recomputing gradients on backward. Defaults to wrapping only `Block`. This saves ~2.6x model parameter size's worth of vram."""

    state_dict_type: Literal["full", "sharded"] = "full"
    """To shard state dictionary or not. TODO: Check how much vram it saves."""

    cpu_offload: bool = False
    """Offload optimizer states to cpu. TODO: Check how much vram it saves."""

    def __post_init__(self) -> None:
        self._cpu_offload = CPUOffload(offload_params=self.cpu_offload)

    def mixed_precision(self, precision: Literal["bf16-true", "bf16-mixed", "32-true", None]) -> int:
        """Number of iterations to warm up the learning rate."""
        if precision == "bf16-mixed":
            return MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32)

        if precision == "bf16-true":
            return MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)

        # FSDP with mixed_precision=None defaults to fp32
        return None
