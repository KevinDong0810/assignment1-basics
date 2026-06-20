import json
import regex as re
import tiktoken
from collections import defaultdict
from .common import gpt2_bytes_to_unicode

class BPETokenizer(object):

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        if vocab is not None and merges is not None:
            self.init(vocab, merges, special_tokens)

    def init(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens is not None else []

        self._decode_vocab = vocab
        self._encode_vocab = {}
        for key, value in vocab.items():
            self._encode_vocab[value] = key
        self._merges_dict = {}
        for index, bytes_tuple in enumerate(merges):
            self._merges_dict[bytes_tuple] = index
        self._merge_list_length = len(merges)
        self._encode_cache = {}
        self._temp = 0


    def _pre_tokenizer(self, text: str):
        if len(self.special_tokens) > 0:
            split_pattern = "(" + "|".join([re.escape(token) for token in self.special_tokens]) + ")"
            split_texts = re.split(split_pattern, text)
        else:
            split_texts = [text]
        # print(f"split text: {split_texts}")
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for text in split_texts:
            if text in self.special_tokens:
                # print(f"special text {text}")
                yield text
            else:
                match_iter = re.finditer(PAT, text)
                for word in match_iter:
                    word_s = word.group()
                    yield word_s

    def encode_iterable(self, iterable):
        for text in iterable:
            # print(f"text: {text}")
            for split_token in self._pre_tokenizer(text):
                if split_token in self.special_tokens:
                    yield self._encode_vocab[split_token.encode("utf-8")]
                else:
                    result = self.encode_word(split_token)
                    for token_id in result:
                        yield token_id

    def encode(self, text: str):
        result = []
        for word_s in self._pre_tokenizer(text):
            if word_s in self.special_tokens:
                result.append(self._encode_vocab[word_s.encode("utf-8")])
            else:
                encode_res = self.encode_word(word_s)
                result.extend(encode_res)
        
        return result
    
    def decode(self, ids: list[int]):
        bytes_results = b''.join([self._decode_vocab[v] for v in ids])
        return bytes_results.decode("utf-8", errors='replace')

    def encode_word(self, word_str: str):
        if word_str in self._encode_cache:
            return self._encode_cache[word_str]
        res = self._encode_word(word_str.encode("utf-8"))
        self._encode_cache[word_str] = res
        return res

    def _encode_word(self, pre_token: bytes):
        token_list = tuple([bytes([b]) for b in pre_token])
        
        # build initial stats-like map;
        priority_map = {}
        indices_map = defaultdict(int)
        for i in range(len(token_list) - 1):
            comb_tuple = token_list[i:i+2]
            priority = self._merges_dict.get(comb_tuple, self._merge_list_length)
            priority_map[comb_tuple] = priority
            indices_map[comb_tuple] += 1

        if len(token_list) > 1:
            comb_tuple, priority = self._get_smallest(priority_map)
            while priority < self._merge_list_length:
                # print(f"current update tuple {comb_tuple} old token list {token_list}")
                token_list = self.replace_tuple(comb_tuple, priority_map, token_list, indices_map)
                # print(f"new token list {token_list}")
                # print(f"priority map {priority_map}")
                comb_tuple, priority = self._get_smallest(priority_map)
        
        result = [self._encode_vocab[b] for b in token_list]
        return result  # list of int
    
    def from_file(self, vocab_filepath, merges_filepath, special_tokens=None):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        
        self.init(vocab, merges, special_tokens)


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
    
    def replace_tuple(self, update_tuple, priority_map, token_list, indices_map):
        first, second = update_tuple
        new_vocab = first + second
        priority_map[update_tuple] = self._merge_list_length  # disable this match
        new_token_list = self.generate_new_word(token_list, update_tuple)
        old_word_length = len(token_list)
        i = 0
        while True:
            try: 
                i = token_list.index(first, i)
            except ValueError:
                break
            if i + 1 < old_word_length and token_list[i + 1] == second:
                if i > 0: # 处理左邻居
                    old_tuple = token_list[i - 1 : i + 1]
                    indices_map[old_tuple] -= 1
                    if old_tuple in priority_map and indices_map[old_tuple] < 1:
                        priority_map[old_tuple] = self._merge_list_length  # remove from possible merges
                if i < old_word_length - 2:
                    # 要排除掉右边也是pair的情况
                    if not ( i < old_word_length - 3 and token_list[i + 2] == first and token_list[i + 3] == second ):
                        # 此时左邻居会处理，因此跳过
                        old_tuple = token_list[i+1: i + 3]
                        indices_map[old_tuple] -= 1
                        if old_tuple in priority_map and indices_map[old_tuple] < 1:
                            priority_map[old_tuple] = self._merge_list_length  # remove from possible merges
                i += 2
            else:
                i += 1
        
                    # 添加新的tuple:
        i = 0
        while True:
            try:
                i = new_token_list.index(new_vocab, i)
            except ValueError:
                break
            if i > 0:
                new_tuple = new_token_list[i - 1: i + 1]
                priority_map[new_tuple] = self._merges_dict.get(new_tuple, self._merge_list_length)
            if i + 1 < len(new_token_list):
                if new_token_list[i + 1] != new_vocab:
                    new_tuple = new_token_list[i : i + 2]
                    priority_map[new_tuple] = self._merges_dict.get(new_tuple, self._merge_list_length)
            i += 1
        return new_token_list


    def _get_smallest(self, priority_map: dict[tuple, int]):
        min_key =  min(priority_map, key = lambda x : priority_map[x])
        return min_key, priority_map[min_key]


if __name__ == "__main__":

    from .adapters import get_tokenizer
    from .common import FIXTURES_PATH, gpt2_bytes_to_unicode

    VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
    MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

    tokenizer = BPETokenizer(None, None)
    tokenizer.from_file(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])

    corpus_path = FIXTURES_PATH / "tinystories_sample.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    assert all_ids == reference_ids

    assert tokenizer.decode(all_ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents
    
