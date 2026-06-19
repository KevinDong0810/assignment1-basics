import regex as re
import os
from collections import defaultdict
import json
import copy
from multiprocessing import Pool
import heapq
import time
import psutil

def find_chunk_boundaries(
    file,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def worker(args):
    input_path, start, end = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        split_pattern = "|".join([re.escape(token) for token in  ["<|endoftext|>"]  ])
        split_texts = re.split(split_pattern, chunk)

        word_freq_map = defaultdict(int)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for text in split_texts:
            match_iter = re.finditer(PAT, text)
            for word in match_iter:
                word_s = word.group()
                byte_tuple = tuple(bytes([b]) for b in word_s.encode("utf-8"))
                word_freq_map[byte_tuple] += 1
    return word_freq_map


class ReverseBytes:
    __slots__ = ("v",)
    def __init__(self, v): self.v = v
    def __lt__(self, other): return self.v > other.v   # 反转比较
    def __eq__(self, other): return self.v == other.v


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
        self.heap = []

    def create_word_list_mp(self, input_path: str | os.PathLike):
        num_processes = 6
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        with Pool(processes=num_processes) as pool:
            pool_args = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
            results = pool.map(worker, pool_args)
        
        result_map = {}
        for result in results:
            result_map.update(result)
        sorted_word_list = [(word, freq) for word, freq in result_map.items()]
        self.sorted_word_list = sorted(sorted_word_list, key = lambda x : x[1])        

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

        self.heap = [(-freq, ReverseBytes(pair)) for pair, freq in self.stats.items()]
        heapq.heapify(self.heap)
    
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
    
    def get_highest_pair(self):
        while self.heap:
            neg_freq, pair = heapq.heappop(self.heap)
            # 校验是否过期：堆里的频率必须和当前 stats 一致才算有效
            if -neg_freq == self.stats.get(pair.v, 0):
                return pair.v
            # 否则丢弃，继续弹
        return None  # stats 为空

    def _bump_stats(self, pair, delta):
        self.stats[pair] += delta
        heapq.heappush(self.heap, (-self.stats[pair], ReverseBytes(pair)))
    
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
                            self._bump_stats(old_tuple, -word_freq)
                            self.indices[old_tuple][j] -= 1
                    if i < old_word_length - 2:
                        # 要排除掉右边也是pair的情况
                        if not ( i < old_word_length - 3 and old_word[i + 2] == first and old_word[i + 3] == second ):
                            # 此时左邻居会处理，因此跳过
                            old_tuple = old_word[i+1: i + 3]
                            if self.indices[old_tuple][j] > 0:
                                self._bump_stats(old_tuple, -word_freq)
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
                    self._bump_stats(new_tuple, word_freq)
                    self.indices[new_tuple][j] += 1
                if i + 1 < len(new_word):
                    if new_word[i + 1] != new_vocab:
                        new_tuple = new_word[i : i + 2]
                        self._bump_stats(new_tuple, word_freq)
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
    
    def train(self, input_path: str | os.PathLike, is_mp=False):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        t_start = time.time()

        if is_mp:
            self.create_word_list_mp(input_path)
        else:
            self.create_word_list(input_path)
        self.create_pair_stats(self.sorted_word_list)

        self.merge_list = []
        total_merges = self.vocab_size - len(self.vocab)
        t_loop_start = time.time()
        while len(self.vocab) < self.vocab_size:
            update_tuple = self.get_highest_pair()
            self.merge_list.append(update_tuple)
            changes = self.replace_pair(update_tuple)
            self.update_frequency(update_tuple, changes)

            done = len(self.merge_list)
            if done % 100 == 0 or done == total_merges:
                elapsed_loop = time.time() - t_loop_start
                eta = elapsed_loop / done * (total_merges - done)
                print(f"\r[{done}/{total_merges}] ETA: {eta:.0f}s", end="", flush=True)
        print()

        elapsed = time.time() - t_start
        mem_used = (process.memory_info().rss - mem_before) / 1024 / 1024
        print(f"Training time: {elapsed:.2f}s | Memory used: {mem_used:.1f} MB")
    
    def show_map(self, input_map: dict, map_name=None):
        if map_name is not None:
            print(f"dict name {map_name}")
        for key, value in input_map.items():
            print(f"key: {key} value {value}")

    def get_vocab(self):
        return self.vocab
    
    def get_merge(self):
        return self.merge_list
    
    def write_results(self, output_dir):
        from tests.common import gpt2_bytes_to_unicode
        byte_encoder = gpt2_bytes_to_unicode()
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as f:
            for token1, token2 in self.merge_list:
                s1 = "".join(byte_encoder[b] for b in token1)
                s2 = "".join(byte_encoder[b] for b in token2)
                f.write(f"{s1} {s2}\n")

        with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
            vocab_serializable = {
                idx: "".join(byte_encoder[b] for b in token)
                for idx, token in self.vocab.items()
            }
            json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import sys
    import json
    input_path = sys.argv[1]
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    trainer = BPETrainer(vocab_size, special_tokens)
    trainer.train(input_path, True)
    trainer.write_results("output/owt")
