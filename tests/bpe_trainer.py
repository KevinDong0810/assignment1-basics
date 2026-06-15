import re
import os
from collections import defaultdict


class BPETrainer(object):

    def __init__(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str],):

        # init vocab
        self.vocab = {}
        for i in range(0, 256):
            self.vocab[i] = bytes([i])
        offset = 256
        for token in special_tokens:
            self.vocab[offset] = bytes(token, "utf-8")
            offset += 1

        # split on special tokens
        with open(input_path, "r", encoding="utf-8") as f:
            orig_text = f.read()
        split_pattern = "|".join([re.escape(token) for token in special_tokens])
        split_texts = re.split(split_pattern, orig_text)

        # frequency count
        self.word_freq_map = defaultdict(int)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for text in split_texts:
            match_iter = re.finditer(PAT, text)
            for word in match_iter:
                self.word_freq_map[word.encode("utf-8")] += 1

        pair_freq_map, pair_set_map = self.build_freq_map()
        
        while len(self.vocab) < vocab_size:
            update_tuple = self.get_highest_pair(pair_freq_map)
            self.update_frequency(update_tuple, pair_freq_map, pair_set_map)
        
    def get_vocab(self):
        return self.vocab
    
    def build_freq_map(self):
        # build initial merge dict
        pair_freq_map = defaultdict(int)
        pair_set_map = defaultdict(set)
        vocab_keys = list(self.vocab.keys())
        vocab_size = len(vocab_keys)

        for i in range(vocab_size):
            for j in range(i, vocab_size):
                query_word = self.vocab[vocab_keys[i]] + self.vocab[vocab_keys[j]]
                query_tuple = (self.vocab[vocab_keys[i]], self.vocab[vocab_keys[j]])
                for key_word, freq in self.word_freq_map.items():
                    if query_word in key_word:
                        # found a match
                        pair_freq_map[query_tuple] += freq
                        pair_set_map[query_tuple].add(key_word)
        
        return pair_freq_map, pair_set_map

    def get_highest_pair(self, pair_freq_mcap: dict):
        return max(pair_freq_mcap, key=lambda k: (pair_freq_mcap[k], k[0] + k[1]))
    
    def update_frequency(self, target_tuple: tuple, pair_freq_map: dict, pair_set_map: dict):
        # update vocab dict
        index = len(self.vocab)
        self.vocab[index] = "".join(target_tuple)
        
        # delete tuple from related maps
        del pair_freq_map[target_tuple]
        del pair_set_map[target_tuple]
        
        # search in the whole pair map, update overlaped tuples
        for tuple in pair_freq_map.keys():
            if tuple[0] != target_tuple[1] and tuple[1] != target_tuple[0]:
                continue
            if tuple[1] == target_tuple[0]:
                new_tuple = tuple[0] + target_tuple[0] + target_tuple[1]
                new_tuple_key = (tuple[0], target_tuple[0] + target_tuple[1])
            else:
                new_tuple = target_tuple[0] + target_tuple[1] + tuple[1]
                new_tuple_key = (target_tuple[0] + target_tuple[1], tuple[1])
            to_be_pop_set = []
            for word in pair_set_map[tuple]:
                if new_tuple in word:
                    freq = self.word_freq_map[word]
                    pair_freq_map[new_tuple_key] += freq
                    pair_freq_map[tuple] -= freq
                    to_be_pop_set.append(word)
            pair_set_map[tuple].difference_update(to_be_pop_set)
            pair_set_map[new_tuple_key].update(to_be_pop_set)

