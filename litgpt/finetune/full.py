# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import dataclasses
import math
import os
import time
from pathlib import Path
import pprint
from typing import Dict, List, Literal, Optional, Tuple, Union

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import RunningMean

from litgpt.args import EvalArgs, LogArgs, TrainArgs, FSDPArgs
from litgpt.data import Alpaca, DataModule, HTXSUTDFinetune
from litgpt.generate.base import generate
from litgpt.model import GPT, Block, Config
from litgpt.prompts import save_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    CycleIterator,
    auto_download_checkpoint,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    create_finetuning_performance_report,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    load_checkpoint,
    num_parameters,
    parse_devices,
    save_hyperparameters,
    select_sft_generate_example,
    capture_hparams,
)


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path("out/finetune/full"),
    precision: Optional[str] = None,
    devices: Union[int, str] = 1,
    num_nodes: int = 1,
    resume: Union[bool, Literal["auto"], Path] = False,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=16,
        micro_batch_size=1,
        lr_warmup_steps=100,
        epochs=5,
        max_seq_length=None,
    ),
    eval: EvalArgs = EvalArgs(interval=600, max_new_tokens=100, max_iters=100),
    log: LogArgs = LogArgs(),
    optimizer: Union[str, Dict] = "AdamW",
    logger_name: Literal["wandb", "tensorboard", "csv", "mlflow"] = "csv",
    seed: int = 1337,
    access_token: Optional[str] = None,
    float32_matmul_precision: Literal["highest", "high", "medium"] = "high",
    pad_id: Optional[int] = None,
    fsdp: FSDPArgs = FSDPArgs(),
) -> None:
    """Finetune a model.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        devices: How many devices/GPUs to use
        num_nodes: How many nodes the code is being run on.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
            ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
        float32_matmul_precision: Optimise matmul. By default this is "highest", whereas "high" trades off some precision with more speed. See: https://pytorch.org/docs/stab
        pad_id: Specify the pad token, as it is often missing from tokenizers.
    """

    # float32_matmul_precision: Optimise matmul. By default this is "highest", whereas "high" trades off some precision with more speed. See: https://pytorch.org/docs/stab
    torch.set_float32_matmul_precision(float32_matmul_precision)

    checkpoint_dir = auto_download_checkpoint(model_name=checkpoint_dir, access_token=access_token)
    hparams = capture_hparams()
    # data = Alpaca() if data is None else data
    data = HTXSUTDFinetune() if data is None else data
    data.seed = seed
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        resume=bool(resume),
        log_interval=train.log_interval,
        log_args=dataclasses.asdict(log),
    )

    if devices * num_nodes > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            state_dict_type=fsdp.state_dict_type,
            sharding_strategy=fsdp.sharding_strategy,
            activation_checkpointing_policy={Block} if fsdp.activation_checkpointing else None,
            mixed_precision=fsdp.mixed_precision(precision),
            # cpu_offload=fsdp._cpu_offload, # TODO: Not working.
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, num_nodes=num_nodes, strategy=strategy, precision=precision, loggers=logger)

    tokenizer = Tokenizer(checkpoint_dir)

    # PAD_ID -- we need this block of code because pad_ids in tokenizers are often broken.

    # if no tokenizer, it could mean that the data is already tokenized. However, we still need pad_id
    if tokenizer is None and pad_id is None:
        raise ValueError("Need to specify `pad_id`, or provide the tokenizer for me to obtain from there.")

    if tokenizer and tokenizer.pad_id is None and pad_id is None:
        raise ValueError("Need to specify `pad_id` as none found in tokenizer.")

    if tokenizer and isinstance(tokenizer.pad_id, int):
        # If the tokenizer has a pad_id, there is no reason for you to specify one. Most likely a mistake.
        if isinstance(pad_id, int) and tokenizer.pad_id != pad_id:
            raise ValueError(
                f"Pad_id of [{tokenizer.pad_id}] found in tokenizer. But you also specified `pad_id`[{pad_id}]. Remove `pad_id`."
            )

        # If all good, we assign the tokenizer's pad_id to pad_id
        fabric.print(f"Pad token id[{tokenizer.pad_id}] found in tokenizer.")
        pad_id = tokenizer.pad_id

    # If we are using a tokenizer, and it doesn't have a pad_id, we can assign it.
    if tokenizer and tokenizer.pad_id is None:
        fabric.print(f"No pad token found in tokenizer. Assigning `pad_id`[{pad_id}].")
        tokenizer.pad_id = pad_id

    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch(
        main, devices, resume, seed, config, data, checkpoint_dir, out_dir, train, eval, optimizer, tokenizer, num_nodes
    )

    fabric.print(pprint.pformat(hparams))
    if logger_name in ("tensorboard", "wandb", "mlflow"):
        fabric.logger.log_hyperparams(hparams)


def main(
    fabric: L.Fabric,
    devices: int,
    resume: Union[bool, Literal["auto"], Path],
    seed: int,
    config: Config,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: Union[str, Dict],
    tokenizer: Tokenizer,
    num_nodes: int = 1,
) -> None:
    validate_args(train, eval)

    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train)
    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(devices, num_nodes)
    lr_max_steps = min(train.epochs * steps_per_epoch, (train.max_steps or float("inf")))

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")

    model = fabric.setup(model)

    optimizer = instantiate_torch_optimizer(optimizer, model.parameters())
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps)
    state = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "iter_num": 0, "step_count": 0}

    resume = find_resume_path(resume, out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
    else:
        load_checkpoint(fabric, state["model"], checkpoint_path)

    train_time = time.perf_counter()
    token_counts = fit(
        fabric=fabric,
        state=state,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        devices=devices,
        num_nodes=num_nodes,
        resume=resume,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        train=train,
        eval=eval,
        data=data,
    )
    training_time = time.perf_counter() - train_time
    output = create_finetuning_performance_report(training_time, token_counts, fabric.device.type)
    fabric.print(output)

    # Final evaluation
    if eval.final_validation:
        val_loss = validate(fabric, model, val_dataloader, dataclasses.replace(eval, max_iters=len(val_dataloader)))
        metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(f"Final evaluation | val loss: {val_loss.item():.3f} | val ppl: {math.exp(val_loss):.3f}")

    # Save the final checkpoint at the end of training
    save_path = out_dir / "final" / "lit_model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fabric.save(save_path, {"model": state["model"]})
    if fabric.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
        copy_config_files(checkpoint_dir, save_path.parent)
        save_hyperparameters(setup, save_path.parent)
        save_prompt_style(data.prompt_style, save_path.parent)


def fit(
    fabric: L.Fabric,
    state: Dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    resume: Union[bool, Literal["auto"], Path],
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    data: DataModule,
    num_nodes: int = 1,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    tokenizer = Tokenizer(checkpoint_dir)

    if train.max_seq_length:
        fabric.print(
            f"`train.max_seq_length` of {train.max_seq_length} is set."
            f" Skipping step of inspecting dataset for the longest sequence length."
            f" If you want to automatically find optimal max sequence length, unset `train.max_seq_length`."
        )
        longest_seq_length = train.max_seq_length
        model.max_seq_length = train.max_seq_length

    else:
        longest_seq_length, longest_seq_ix = get_longest_seq_length(
            ConcatDataset([train_dataloader.dataset, val_dataloader.dataset])
        )
        model.max_seq_length = min(longest_seq_length, train.max_seq_length or float("inf"))

        fabric.print(
            f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
            f" {model.max_seq_length} and context length is {model.config.block_size}"
        )

    token_counts = {
        "raw_tokens": torch.tensor(0, device=fabric.device, dtype=torch.long),
        "raw_tokens_plus_prompt_template": torch.tensor(0, device=fabric.device, dtype=torch.long),
        "raw_tokens_plus_prompt_template_and_padding": torch.tensor(0, device=fabric.device, dtype=torch.long),
    }

    if eval.initial_validation:
        val_loss = validate(fabric, model, val_dataloader, dataclasses.replace(eval, max_iters=len(val_dataloader)))
        val_loss = f"{val_loss:.3f}"
    else:
        fabric.print("Verifying settings ...")
        validate(fabric, model, val_dataloader, dataclasses.replace(eval, max_iters=2), verbose=False)  # sanity check
        val_loss = "n/a"

    initial_iter = state["iter_num"]
    max_steps = train.max_steps or float("inf")
    train_iterator = CycleIterator(train_dataloader)

    # resume data loader state by fast-forwarding through all seen batches
    if resume:
        resume_t0 = time.perf_counter()
        for resume_iter in range(initial_iter):
            next(train_iterator)
            if resume_iter % 1000 == 0:
                fabric.print(f"Resuming dataset: {resume_iter} / {initial_iter}")
        fabric.barrier()
        fabric.print(
            f"Resuming data loader finished. Took {time.perf_counter() - resume_t0:.1f} seconds to reach iteration"
            f" {initial_iter}."
        )

    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=False).to(
        fabric.device
    )
    fabric.barrier()

    while state["step_count"] < max_steps:
        state["iter_num"] += 1
        iter_t0 = time.perf_counter()
        batch = next(train_iterator)
        if train_iterator.epoch >= train.epochs:
            break
        input_ids, targets = batch["input_ids"], batch["labels"]

        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices, num_nodes) != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            # shift the targets such that output n predicts token n+1
            loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
            fabric.backward(loss / train.gradient_accumulation_iters(devices, num_nodes))

        running_loss.update(loss.detach())

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            state["step_count"] += 1

        token_counts["raw_tokens"] += batch["token_counts"]["raw"].sum().item()
        token_counts["raw_tokens_plus_prompt_template"] += (
            batch["token_counts"]["raw_plus_prompt_template"].sum().item()
        )
        token_counts["raw_tokens_plus_prompt_template_and_padding"] += input_ids.numel()

        if state["iter_num"] % train.log_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "tokens": token_counts["raw_tokens_plus_prompt_template"],
                "total_tokens": token_counts["raw_tokens_plus_prompt_template"] * fabric.world_size,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            if isinstance(val_loss, torch.Tensor):
                val_loss = f"{val_loss:.3f}"
            fabric.print(
                f"Epoch {metrics['epoch'] + 1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
            )
            fabric.log_dict(metrics, step=state["iter_num"])

        if not is_accumulating and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, eval)
            generate_example(fabric, model, tokenizer, eval, data)
            t1 = time.perf_counter() - t0

            val_loss_tensor = val_loss.detach().clone().to(fabric.device)
            val_time_tensor = torch.tensor(t1, device=fabric.device, dtype=torch.float32)

            fabric.all_reduce(val_loss_tensor, reduce_op="mean")
            fabric.all_reduce(val_time_tensor, reduce_op="mean")

            fabric.print(
                f"iter {state['iter_num']}: val loss {val_loss_tensor.item():.4f}, val time: {val_time_tensor.item() * 1000:.2f} ms"
            )
            metrics = {"val_loss": val_loss_tensor, "val_ppl": math.exp(val_loss_tensor)}
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()
        if train.save_interval is not None and not is_accumulating and state["step_count"] % train.save_interval == 0:
            checkpoint_file = out_dir / f"step-{state['step_count']:06d}" / "lit_model.pth"
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            fabric.print(f"Saving checkpoint to {str(checkpoint_file.parent)!r}")
            fabric.save(checkpoint_file, state)
            if fabric.global_rank == 0:
                copy_config_files(checkpoint_dir, checkpoint_file.parent)
                save_hyperparameters(setup, checkpoint_file.parent)
                save_prompt_style(data.prompt_style, checkpoint_file.parent)

    total_token_counts = {}
    for key in token_counts:
        total = fabric.all_reduce(token_counts[key], reduce_op="sum")
        total_token_counts[key] = total.item()

    return total_token_counts


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, val_dataloader: DataLoader, eval: EvalArgs, verbose: bool = True
) -> torch.Tensor:
    if verbose:
        fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)

    val_loss = losses.mean()
    model.train()
    return val_loss


@torch.no_grad()
def generate_example(fabric: L.Fabric, model: GPT, tokenizer: Tokenizer, eval: EvalArgs, data: DataModule):
    instruction = select_sft_generate_example(eval, data)
    fabric.print(instruction)
    prompt = data.prompt_style.apply(instruction)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    model.eval()

    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)

    max_returned_tokens = len(encoded) + eval.max_new_tokens

    if max_returned_tokens < model.max_seq_length:
        with fabric.init_tensor():
            # do not set `max_seq_length=max_returned_token` because memory is not a concern here
            model.set_kv_cache(batch_size=1)
        output = generate(
            model, encoded, max_returned_tokens=max_returned_tokens, temperature=0.8, eos_id=tokenizer.eos_id
        )
        model.clear_kv_cache()
        model.train()
        output = tokenizer.decode(output)
        fabric.print(f"{output}\n")
    else:
        print(
            f"Length of encoded instruction ({len(encoded)}) and eval.max_new_tokens ({eval.max_new_tokens}) "
            f"exceeds model.max_seq_length ({model.max_seq_length}) used for training. Skipping example generation for efficiency. "
            f"The model's supported context size (post-training) is {model.config.block_size}."
        )


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs
) -> Tuple[DataLoader, DataLoader]:
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=train.max_seq_length)
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    return train_dataloader, val_dataloader


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [(train, ["max_tokens", "max_norm", "tie_embeddings", "lr_warmup_fraction"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if not train.epochs and not train.max_steps:
        issues.append(f"{__file__} requires either epochs or max_steps to be set. This is set in {train}")
    if issues:
        raise ValueError("\n".join(issues))


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup, as_positional=False)
