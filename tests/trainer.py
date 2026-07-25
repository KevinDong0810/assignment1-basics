import train_config as config_utlis
import train_module as train_utlis
import net_module as network_utlis

import os
import torch
import yaml
import wandb
import argparse
import time

import numpy as np

from dataclasses import asdict
from datetime import datetime, timedelta
from tqdm import tqdm

def parse_config(path=None):
    default_config = config_utlis.TrainConfig()
    if path is not None:
        with open(path, "r") as fid:
            loaded_config = yaml.safe_load(fid)
        config_utlis.apply_overrides(default_config, loaded_config)
    
    return default_config


def prepare(config: config_utlis.TrainConfig):
    # build model
    network = network_utlis.TransformerLM(
        config.model.vocab_size, config.model.context_length, config.model.d_model,
        config.model.num_layers, config.model.num_heads, config.model.d_ff, config.model.theta).to(config.runtime.device)
    
    optimizer = train_utlis.AdamW(network.parameters(), config.optimizer.lr, config.optimizer.weight_decay,
                                  (config.optimizer.beta1, config.optimizer.beta2), config.optimizer.eps)
    return network, optimizer


def get_lr(cur_it: int, config: config_utlis.TrainConfig):
    return train_utlis.lr_cosine_func(cur_it, config.optimizer.lr, config.optimizer.min_lr, 
                                      config.optimizer.warmup_steps, config.optimizer.cos_steps)


def train(model: network_utlis.TransformerLM, optimizer: train_utlis.AdamW, train_dataset, val_dataset, config: config_utlis.TrainConfig, restore_path=None, exp_name="llm_test"):
    start = 0
    loss_func = train_utlis.CrossEntropyLoss()
    ckpt_dir = os.path.join(config.runtime.checkpoint_path, exp_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if restore_path is not None:
        loaded_steps = train_utlis.load_checkpoint(restore_path, model, optimizer)
        start = loaded_steps + 1

    with wandb.init(project="cs336-assignment1", name=exp_name, config=asdict(config)) as run:
        run.define_metric("global_step")
        run.define_metric("train/*", step_metric="global_step")
        run.define_metric("val/*", step_metric="global_step")

        training_start_time = time.perf_counter()
        completed_steps = 0
        progress = tqdm(
            total=config.runtime.max_steps,
            initial=start,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            mininterval=0,
            miniters=1,
        )

        for step in range(start, config.runtime.max_steps):
            optimizer.zero_grad()
            cur_lr = get_lr(step, config)
            x, y = train_utlis.sample_from_dataset(train_dataset, config.runtime.batch_size, config.model.context_length, device=config.runtime.device)
            logits = model(x)
            loss = loss_func(logits, y)
            loss.backward()
            for param in optimizer.param_groups:
                param["lr"] = cur_lr
            optimizer.step()
            
            if step % config.runtime.checkpoint_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"{step}.ckpt")
                train_utlis.save_checkpoint(model, optimizer, step, ckpt_path)
                progress.write(f"save step {step} to path {ckpt_path}")

            if step % config.runtime.eval_interval == 0:
                with torch.no_grad():
                    out_loss = []
                    for _ in range(50):
                        x, y = train_utlis.sample_from_dataset(val_dataset, config.runtime.batch_size, config.model.context_length, device=config.runtime.device)
                        logits = model(x)
                        val_loss = loss_func(logits, y)
                        out_loss.append(val_loss.item())
                    val_loss = np.mean(out_loss)
                    progress.write(f"step {step} val/loss, {val_loss}")

            if step % config.runtime.log_interval == 0:
                log_dict = {
                    "global_step": step,
                    "train/loss": loss.item(),
                    "train/lr": cur_lr
                }
                progress.write(f"step {step} train/loss, {loss}")
                if step % config.runtime.eval_interval == 0:
                    log_dict["val/loss"] = val_loss
                run.log(log_dict)

            completed_steps += 1
            elapsed_time = time.perf_counter() - training_start_time
            average_step_time = elapsed_time / completed_steps
            remaining_steps = config.runtime.max_steps - step - 1
            estimated_finish_time = datetime.now().astimezone() + timedelta(
                seconds=average_step_time * remaining_steps
            )
            progress.set_postfix_str(
                f"avg={average_step_time:.3f}s/step, "
                f"finish={estimated_finish_time:%Y-%m-%d %H:%M:%S %Z}",
                refresh=False,
            )
            progress.update(1)

            step += 1
        progress.close()

        ckpt_path = os.path.join(ckpt_dir, f"final.ckpt")
        train_utlis.save_checkpoint(model, optimizer, step - 1, ckpt_path)
        print(f"save final model to path {ckpt_path}")
        finished_at = datetime.now().astimezone()
        total_training_time = time.perf_counter() - training_start_time
        average_step_time = (
            total_training_time / completed_steps if completed_steps > 0 else 0.0
        )
        print(
            f"training finished at {finished_at:%Y-%m-%d %H:%M:%S %Z}; "
            f"average time {average_step_time:.3f}s/step"
        )


def create_overfit_token_array(data_length, vocab_size, output_dir):
    output_token = []
    for i in range(data_length):
        output_token.append(i % vocab_size)

    train_path = os.path.join(output_dir, "test_train.npy")
    test_path = os.path.join(output_dir, "test_val.npy")

    np.save(train_path, output_token)
    np.save(test_path, output_token)

def main():

    parser = argparse.ArgumentParser(description="training llm model")
    parser.add_argument("train_dataset_path", type=str)
    parser.add_argument("val_dataset_path", type=str)
    parser.add_argument("--restore_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="llm_test")
    args = parser.parse_args()

    config = parse_config(args.config_path)
    network, optimizer = prepare(config)

    train_dataset = np.load(args.train_dataset_path, mmap_mode="r")
    val_dataset = np.load(args.val_dataset_path, mmap_mode="r")
    train(network, optimizer, train_dataset, val_dataset, config, args.restore_path, args.exp_name)


if __name__ == "__main__":
    # create_overfit_token_array(9, 64, "fixtures")
    main()

