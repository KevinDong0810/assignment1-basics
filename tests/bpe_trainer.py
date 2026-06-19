import regex as re
import os
from collections import defaultdict
import json
import copy


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
        self.word_token_map = defaultdict(list)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for text in split_texts:
            match_iter = re.finditer(PAT, text)
            for word in match_iter:
                word_s = word.group()
                self.word_freq_map[word_s.encode("utf-8")] += 1
        for word in self.word_freq_map.keys():
            token_list = []
            for i in range(len(word)):
                token_list.append(word[i:i+1])
            self.word_token_map[word] = token_list

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
    
    def update_frequency(self, target_tuple: tuple, pair_freq_map: dict, pair_set_map: dict):
        # update vocab dict
        index = len(self.vocab)
        new_vocab = target_tuple[0] + target_tuple[1]
        self.vocab[index] = new_vocab
        # print(f"add new vocab, tuple {target_tuple}, index {index}")
        
        # delete tuple from related maps
        del pair_freq_map[target_tuple]
        related_words = copy.copy(pair_set_map[target_tuple])
        del pair_set_map[target_tuple]
        
        for word in related_words:
            word_token_list = self.word_token_map[word]
            influenced_indexes = []
            influenced_old_tuples = []
            new_generated_tuples = []
            new_token_list = []
            index = 0
            while index < len(word_token_list) - 1:
                if word_token_list[index] == target_tuple[0] and word_token_list[index+1] == target_tuple[1]:
                    influenced_indexes.append(index)
                    index += 2
                else:
                    index += 1
            for i in range(len(word_token_list)):
                if i in influenced_indexes:
                    influenced_old_tuples.append((word_token_list[i], word_token_list[i+1]))
                else:
                    if i + 1 in influenced_indexes:
                        influenced_old_tuples.append((word_token_list[i], word_token_list[i+1]))
                    elif i - 1 in influenced_indexes and i + 1 < len(word_token_list):
                        influenced_old_tuples.append((word_token_list[i], word_token_list[i+1]))
                    
            
            index = 0
            while index < len(word_token_list):
                if index in influenced_indexes:
                    new_token_list.append(new_vocab)
                    index += 2
                else:
                    new_token_list.append(word_token_list[index])
                    index += 1
            
            # 替换整个 new_generated_tuples 生成部分
            new_generated_tuples = []
            for i in range(len(new_token_list) - 1):
                new_generated_tuples.append((new_token_list[i], new_token_list[i + 1]))
            # 然后只保留涉及 new_vocab 的那些（其余未变化的 pair 不需要 +freq）
            new_generated_tuples = [
                p for p in new_generated_tuples
                if p[0] == new_vocab or p[1] == new_vocab
]

            
            self.word_token_map[word] = new_token_list
            for old_tuple in influenced_old_tuples:
                if old_tuple in pair_freq_map:
                    pair_freq_map[old_tuple] -= self.word_freq_map[word]
                    pair_set_map[old_tuple].discard(word)
            for new_tuple in new_generated_tuples:
                pair_freq_map[new_tuple] += self.word_freq_map[word]
                pair_set_map[new_tuple].add(word)
                    

    
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
