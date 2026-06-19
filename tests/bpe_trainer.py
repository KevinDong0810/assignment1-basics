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
    
    def create_word_list(self, input_path: str | os.PathLike):
        # split on special tokens
        with open(input_path, "r", encoding="utf-8") as f:
            orig_text = f.read()
        split_pattern = "|".join([re.escape(token) for token in self.special_tokens])
        split_texts = re.split(split_pattern, orig_text)

        # frequency count
        word_freq_map = defaultdict(int)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for text in split_texts:
            match_iter = re.finditer(PAT, text)
            for word in match_iter:
                word_s = word.group()
                byte_tuple = tuple(bytes([b]) for b in word_s.encode("utf-8"))
                word_freq_map[byte_tuple] += 1
        sorted_word_list = [(word, freq) for word, freq in word_freq_map.items()]
        self.sorted_word_list = sorted(sorted_word_list, key = lambda x : x[1])
        
    def create_pair_stats(self, sorted_word_list: list[tuple[bytes, int]]):
        self.stats = defaultdict(int)  # 记录每个二元组合的频率
        self.indices = defaultdict(lambda: defaultdict(int))  # pair, index of word, pair freq in this word

        for j, (word, word_freq) in enumerate(sorted_word_list):
            for i in range(len(word) - 1):
                pair = tuple(word[i:i + 2])
                self.stats[pair] += word_freq
                self.indices[pair][j] += 1
    
    def replace_pair(self, pair: tuple):
        """ 根据输入pair找到所有受影响的其余pair """
        first, second = pair
        new_pair = first + second
        self.stats[pair] = 0
        self.vocab[len(self.vocab)] = new_pair

        changes = []
        for j, pair_freq in self.indices[pair].items():
            if pair_freq < 1:
                continue
            old_word = self.sorted_word_list[j][0]
            word_freq = self.sorted_word_list[j][1]
            new_word = self.generate_new_word(old_word, pair)
            changes.append((j, old_word, new_word, word_freq))
            self.sorted_word_list[j] = (new_word, word_freq)
        self.indices[pair] = defaultdict(int)

        return changes
    
    def update_frequency(self, pair: tuple, changes:list):
        
        first, second = pair
        new_vocab = first + second
        for j, old_word, new_word, word_freq in changes:
            # 先处理受影响的旧tuple:
            old_word_length = len(old_word)
            i = 0
            while True:
                try: 
                    i = old_word.index(first, i)
                except ValueError:
                    break
                if i + 1 < old_word_length and old_word[i + 1] == second:
                    if i > 0: # 处理左邻居
                        old_tuple = old_word[i - 1 : i + 1]
                        if self.indices[old_tuple][j] > 0:
                            self.stats[old_tuple] -= word_freq
                            self.indices[old_tuple][j] -= 1
                    if i < old_word_length - 2:
                        # 要排除掉右边也是pair的情况
                        if not ( i < old_word_length - 3 and old_word[i + 2] == first and old_word[i + 3] == second ):
                            # 此时左邻居会处理，因此跳过
                            old_tuple = old_word[i+1: i + 3]
                            if self.indices[old_tuple][j] > 0:
                                self.stats[old_tuple] -= word_freq
                                self.indices[old_tuple][j] -= 1
                    i += 2
                else:
                    i += 1

            # 添加新的tuple:
            i = 0
            while True:
                try:
                    i = new_word.index(new_vocab, i)
                except:
                    break
                if i > 0:
                    new_tuple = new_word[i - 1: i + 1]
                    self.stats[new_tuple] += word_freq
                    self.indices[new_tuple][j] += 1
                if i + 1 < len(new_word):
                    if new_word[i + 1] != new_vocab:
                        new_tuple = new_word[i : i + 2]
                        self.stats[new_tuple] += word_freq
                        self.indices[new_tuple][j] += 1
                i += 1                      
        
    def generate_new_word(self, old_word, pair):
        first, second = pair
        joined_pair = first + second
        new_word = []
        index = 0
        while index < len(old_word):
            if old_word[index] == first and index < len(old_word) - 1 and old_word[index + 1] == second:
                new_word.append(joined_pair)
                index += 2
            else:
                new_word.append(old_word[index])
                index += 1
        return tuple(new_word)
    
    def train(self, input_path: str | os.PathLike):
        self.create_word_list(input_path)
        self.create_pair_stats(self.sorted_word_list)
        
        self.merge_list = []
        while len(self.vocab) < self.vocab_size:
            # print(f"============= {len(self.vocab) - 256} iteration ======================")
            update_tuple = self.get_highest_pair(self.stats)
            self.merge_list.append(update_tuple)
            changes = self.replace_pair(update_tuple)
            self.update_frequency(update_tuple, changes)
    
    def show_map(self, input_map: dict, map_name=None):
        if map_name is not None:
            print(f"dict name {map_name}")
        for key, value in input_map.items():
            print(f"key: {key} value {value}")

    def get_vocab(self):
        return self.vocab
    
    def get_merge(self):
        return self.merge_list

    def get_highest_pair(self, pair_freq_mcap: dict):
        return max(pair_freq_mcap, key=lambda k: (pair_freq_mcap[k], k))
    
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
