import regex as re
import os
from collections import defaultdict
import json


class BPETrainer(object):

    def __init__(self, vocab_size: int, special_tokens: list[str],):

        # init vocab
        self.vocab = {}
        for i in range(0, 256):
            self.vocab[i] = bytes([i])
        offset = 256
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        for token in self.special_tokens:
            self.vocab[offset] = bytes(token, "utf-8")
            offset += 1
    
    def train(self, input_path: str | os.PathLike):
        # split on special tokens
        with open(input_path, "r", encoding="utf-8") as f:
            orig_text = f.read()
        split_pattern = "|".join([re.escape(token) for token in self.special_tokens])
        split_texts = re.split(split_pattern, orig_text)

        # frequency count
        self.word_freq_map = defaultdict(int)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for text in split_texts:
            match_iter = re.finditer(PAT, text)
            for word in match_iter:
                word_s = word.group()
                self.word_freq_map[word_s.encode("utf-8")] += 1

        pair_freq_map, pair_set_map = self.build_freq_map()
        
        self.merge_list = []
        while len(self.vocab) < self.vocab_size:
            # print(f"============= {len(self.vocab) - 256} iteration ======================")
            update_tuple = self.get_highest_pair(pair_freq_map)
            self.merge_list.append(update_tuple)
            self.update_frequency(update_tuple, pair_freq_map, pair_set_map)
    
    def show_map(self, input_map: dict, map_name=None):
        if map_name is not None:
            print(f"dict name {map_name}")
        for key, value in input_map.items():
            print(f"key: {key} value {value}")

    def get_vocab(self):
        return self.vocab
    
    def get_merge(self):
        return self.merge_list
    
    def build_freq_map(self):
        # build initial merge dict
        pair_freq_map = defaultdict(int)
        pair_set_map = defaultdict(set)
        vocab_keys = list(self.vocab.keys())
        vocab_size = len(vocab_keys)

        for key_word, freq in self.word_freq_map.items():
            for index in range(len(key_word) - 1):
                a = key_word[index:index + 1]
                b = key_word[index + 1:index + 2]
                query_tuple = (a, b)
                pair_freq_map[query_tuple] += freq
                pair_set_map[query_tuple].add(key_word)
        
        return pair_freq_map, pair_set_map

    def get_highest_pair(self, pair_freq_mcap: dict):
        return max(pair_freq_mcap, key=lambda k: (pair_freq_mcap[k], k))
    
    def update_helper(self, pair_freq_map, pair_set_map, old_tuple, new_tuple, new_tuple_key, to_be_pop_set):
        hit_words = []
        for word in pair_set_map[old_tuple]:
            if new_tuple in word:
                freq = self.word_freq_map[word]
                pair_freq_map[new_tuple_key] += freq
                pair_freq_map[old_tuple] -= freq
                to_be_pop_set.append(word)
                hit_words.append(word)
        pair_set_map[new_tuple_key].update(hit_words)
    
    def update_frequency(self, target_tuple: tuple, pair_freq_map: dict, pair_set_map: dict):
        # update vocab dict
        index = len(self.vocab)
        self.vocab[index] = target_tuple[0] + target_tuple[1]
        # print(f"add new vocab, tuple {target_tuple}, index {index}")
        
        # delete tuple from related maps
        del pair_freq_map[target_tuple]
        del pair_set_map[target_tuple]
        
        # search in the whole pair map, update overlaped tuples
        current_pair_keys = list(pair_freq_map.keys())
        for old_tuple in current_pair_keys:
            if old_tuple[0] != target_tuple[1] and old_tuple[1] != target_tuple[0]:
                continue
            to_be_pop_set = []
            if old_tuple[1] == target_tuple[0]:
                new_tuple = old_tuple[0] + target_tuple[0] + target_tuple[1]
                new_tuple_key = (old_tuple[0], target_tuple[0] + target_tuple[1])
                self.update_helper(pair_freq_map, pair_set_map, old_tuple, new_tuple, new_tuple_key, to_be_pop_set)
            if old_tuple[0] == target_tuple[1]:
                new_tuple = target_tuple[0] + target_tuple[1] + old_tuple[1]
                new_tuple_key = (target_tuple[0] + target_tuple[1], old_tuple[1])
                self.update_helper(pair_freq_map, pair_set_map, old_tuple, new_tuple, new_tuple_key, to_be_pop_set)
            
            pair_set_map[old_tuple].difference_update(to_be_pop_set)

    
    def write_merge(self, output_file):
        from tests.common import gpt2_bytes_to_unicode
        byte_encoder = gpt2_bytes_to_unicode()
        with open(output_file, "w", encoding="utf-8") as f:
            for token1, token2 in self.merge_list:
                s1 = "".join(byte_encoder[b] for b in token1)
                s2 = "".join(byte_encoder[b] for b in token2)
                f.write(f"{s1} {s2}\n")

if __name__ == "__main__":
    import sys
    import json
    input_path = sys.argv[1]
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]

    trainer = BPETrainer(vocab_size, special_tokens)
    trainer.train(input_path)
    trainer.write_merge("merge_output.json")
