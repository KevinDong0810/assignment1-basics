import train_config as config_utlis
import train_module as train_utlis
import net_module as network_utlis
from tests.bpe_tokenizer import BPETokenizer

import os
import torch
import yaml

import argparse


def parse_config(path=None):
    default_config = config_utlis.TrainConfig()
    if path is not None:
        with open(path, "r") as fid:
            loaded_config = yaml.safe_load(fid)
        config_utlis.apply_overrides(default_config, loaded_config)
    
    return default_config


def load_model(config: config_utlis.TrainConfig, checkpoint_path):
    # build model
    network = network_utlis.TransformerLM(
        config.model.vocab_size, config.model.context_length, config.model.d_model,
        config.model.num_layers, config.model.num_heads, config.model.d_ff, config.model.theta).to(config.runtime.device)
    
    optimizer = train_utlis.AdamW(network.parameters(), config.optimizer.lr, config.optimizer.weight_decay,
                                  (config.optimizer.beta1, config.optimizer.beta2), config.optimizer.eps)

    train_utlis.load_checkpoint(checkpoint_path, network, optimizer)
    network.eval()

    return network


def create_decoder(network: network_utlis.TransformerLM, tokenizer_path: str):
    merge_file = os.path.join(tokenizer_path, "merges.txt")
    vocab_file = os.path.join(tokenizer_path, "vocab.json")

    tokenizer = BPETokenizer(None, None)
    tokenizer.from_file(vocab_file, merge_file, ["<|endoftext|>"])

    decoder = network_utlis.LLMDecoder(network, tokenizer)
    return decoder


def generate_output(
    decoder: network_utlis.LLMDecoder,
    maximum_length: int,
    temperature: float,
    top_p: float,
):
    print("Interactive inference is ready.")
    print("Enter a single-line prompt, or use /exit or /quit to stop.")

    while True:
        try:
            prompt = input("\nPrompt> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if prompt.strip().lower() in {"/exit", "/quit"}:
            print("Exiting.")
            break
        if not prompt:
            continue

        prompt_length = len(decoder.tokenizer.encode(prompt))
        context_length = decoder.llm_module.context_length
        if prompt_length >= context_length:
            print(
                f"Prompt is too long: {prompt_length} tokens; "
                f"the model context length is {context_length}."
            )
            continue

        with torch.inference_mode():
            decoded_text = decoder.generate(
                prompt,
                maximum_length=maximum_length,
                temperature=temperature,
                top_p=top_p,
            )

        generated_text = decoded_text[len(prompt):]
        print(f"\nModel> {generated_text}", flush=True)

def main():

    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("tokenizer_path", type=str)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be greater than 0")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")
    if not 0.0 <= args.top_p <= 1.0:
        parser.error("--top_p must be between 0 and 1")

    config = parse_config(args.config_path)
    network = load_model(config, args.checkpoint)
    decoder = create_decoder(network, args.tokenizer_path)
    generate_output(
        decoder,
        maximum_length=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
