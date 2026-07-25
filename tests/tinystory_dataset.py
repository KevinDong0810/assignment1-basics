from tests.bpe_tokenizer import BPETokenizer

import os
import numpy as np
from tqdm import tqdm


def iter_lines_with_progress(fid, file_path, description):
    total_bytes = os.path.getsize(file_path)
    with tqdm(
        total=total_bytes,
        desc=description,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    ) as progress:
        for line in fid:
            yield line
            progress.update(len(line.encode("utf-8")))


def create_dataset(train_txt_path, val_txt_path, output_dir, tokenizer: BPETokenizer):
    with open(train_txt_path, "r", encoding="utf-8", newline="") as fid:
        train_ids = []
        train_lines = iter_lines_with_progress(fid, train_txt_path, "Encoding train")
        train_encoded_iter = tokenizer.encode_iterable(train_lines)
        for result in train_encoded_iter:
            train_ids.append(result)

    train_path = os.path.join(output_dir, "tiny_story_train.npy")
    np.save(train_path, train_ids)

    with open(val_txt_path, "r", encoding="utf-8", newline="") as fid:
        val_ids = []
        val_lines = iter_lines_with_progress(fid, val_txt_path, "Encoding validation")
        val_encoded_iter = tokenizer.encode_iterable(val_lines)
        for result in val_encoded_iter:
            val_ids.append(result)

    val_path = os.path.join(output_dir, "tiny_story_val.npy")
    np.save(val_path, val_ids)

    print(f"write train and val to dir {output_dir}")


if __name__ == "__main__":
    train_txt_path = "data/TinyStoriesV2-GPT4-train.txt"
    val_txt_path = "data/TinyStoriesV2-GPT4-valid.txt"
    output_dir = "data"

    vocab_path = "output/TinyStories/vocab.json"
    merge_path = "output/TinyStories/merges.txt"
    tokenizer = BPETokenizer(None, None, None)
    tokenizer.from_file(vocab_path, merge_path, ["<|endoftext|>"])
    create_dataset(train_txt_path, val_txt_path, output_dir, tokenizer)


    
