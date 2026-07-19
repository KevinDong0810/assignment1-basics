import train_config as config_utlis
import train_module as train_utlis
import net_module as network_utlis

import os
import torch
import numpy as np
import yaml

import argparse

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
        config.model.num_layers, config.model.num_heads, config.model.d_ff, config.model.theta)
    
    optimizer = train_utlis.AdamW(network.parameters(), config.optimizer.lr, config.optimizer.weight_decay,
                                  (config.optimizer.beta1, config.optimizer.beta2), config.optimizer.eps)
    return network, optimizer


def train(model: network_utlis.TransformerLM, optimizer: train_utlis.AdamW, train_dataset, val_dataset, config: config_utlis.TrainConfig, restore_path=None, exp_name="llm_test"):
    start = 0
    loss_func = train_utlis.CrossEntropyLoss()
    ckpt_dir = os.path.join(config.runtime.checkpoint_path, exp_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if restore_path is not None:
        loaded_steps = train_utlis.load_checkpoint(restore_path, model, optimizer)
        start = loaded_steps + 1
    
    for step in range(start, config.runtime.max_steps):
        x, y = train_utlis.sample_from_dataset(train_dataset, config.runtime.batch_size, config.model.context_length, device=config.runtime.device)
        logits = model(x)
        loss = loss_func(logits, y)
        loss.backward()
        optimizer.step()

        if step % config.runtime.log_interval == 0:
            print(f"step {step} train/loss, {loss}")
        
        if step % config.runtime.checkpoint_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{step}.ckpt")
            train_utlis.save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"save step {step} to path {ckpt_path}")

        if step % config.runtime.eval_interval == 0:
            with torch.no_grad():
                x, y = train_utlis.sample_from_dataset(val_dataset, config.runtime.batch_size, config.model.context_length, device=config.runtime.device)
                logits = model(x)
                val_loss = loss_func(logits, y)
                print(f"step {step} val/loss, {val_loss}")
        
        step += 1
        ckpt_path = os.path.join(ckpt_dir, f"final.ckpt")
        train_utlis.save_checkpoint(model, optimizer, step, ckpt_path)
        print(f"save final model to path {ckpt_path}")


def main():

    parser = argparse.ArgumentParser(description="training llm model")
    parser.add_argument("train_dataset_path", type="str", required=True)
    parser.add_argument("val_dataset_path", type="str", required=True)
    parser.add_argument("restore_path", type="str", default=None)
    parser.add_argument("config_path", type="str", default=None)
    parser.add_argument("exp_name", type="str", default="llm_test")
    args = parser.parse_args()

    config = parse_config(args.config_path)
    network, optimizer = prepare(config)

    train_dataset = np.load(args.train_dataset_path, mmap_mode="r")
    val_dataset = np.load(args.val_dataset_path, mmap_mode="r")
    train(network, optimizer, train_dataset, val_dataset, config, args.restore_path, args.exp_name)



