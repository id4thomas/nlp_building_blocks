'''
Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE
tokenizer. Your BPE training function should handle (at least) the following input parameters:
input_path: str Path to a text file with BPE tokenizer training data.

vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
initial byte vocabulary, vocabulary items produced from merging, and any special tokens).

special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
otherwise affect BPE training.

Your BPE training function should return the resulting vocabulary and merges:
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
lary) to bytes (token bytes).

merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
<token2>. The merges should be ordered by order of creation.

To test your BPE training function against our provided tests, you will first need to implement the
test adapter at [adapters.run_train_bpe]. Then, run uv run pytest tests/test_train_bpe.py.
Your implementation should be able to pass all tests. Optionally (this could be a large time-investment),
you can implement the key parts of your training method using some systems language, for instance
C++ (consider cppyy for this) or Rust (using PyO3). If you do this, be aware of which operations
require copying vs reading directly from Python memory, and make sure to leave build instructions, or
make sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supported
in most regex engines and will be too slow in most that do. We have verified that Oniguruma is
reasonably fast and supports negative lookahead, but the regex package in Python is, if anything,
even faster.

# Naive Version
* All sequential

'''

import multiprocessing

from collections import defaultdict
import os
import regex as re
from typing import List, Dict, Tuple
 
from typing import BinaryIO

# GPT-2 Pretokenization pattern
PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
WHITESPACE_TOKEN_BYTES="Ġ".encode('utf-8')
NEWLINE_TOKEN_BYTES="Ċ".encode('utf-8')

full_token_re = re.compile(rf"^(?:{PAT})$")


class TokenNode:
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None
        # For determining pre-tokenization boundary
        self.is_next_connected = True
        self.span = None
        
def add_node(byte_val, prev, span=None):
    """Helper to create and link a new TokenNode."""
    node = TokenNode(byte_val)
    node.span=span
    if prev:
        prev.next = node
        node.prev = prev
    return node


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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

def pretokenize(chunk, start_pos: int = 0) -> Tuple[TokenNode, TokenNode]:
    head=None
    prev=None

    # Outer: Pre-tokenized Tokens
    for pre_tok in re.finditer(PAT, chunk):
        text = pre_tok.group()
        span = pre_tok.span()
        span = (start_pos+span[0], start_pos+span[1])
        bytes_to_process = text.encode('utf-8')

        # Add remaining bytes as separate nodes
        for byte in bytes_to_process:
            prev = add_node(bytes([byte]), prev, span=span)
            if head is None:
                head = prev

        if prev:
            prev.is_next_connected = False
    return head, prev

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = 8,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    '''
    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    '''
    # 1. Initialize Vocab
    vocab: Dict[int, bytes] = {}
    cur_vocab_size = 0
    
    ## 1-1. Special tokens
    split_pat = re.compile(
        "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
    )
    encoded_special_tokens = [x.encode('utf-8') for x in special_tokens]
    for tok in encoded_special_tokens:
        vocab[cur_vocab_size]=tok
        cur_vocab_size+=1
    split_special_token = "<|endoftext|>".encode('utf-8')
        
    ## 1-1. 256 utf-8 bytes
    # byte can represent 256 values (unicode string is sequence of bytes)
    # start with single-byte -> merge
    for i in range(256):
        vocab[cur_vocab_size]=bytes([i])
        cur_vocab_size+=1
    
    # 2. Pre-Tokenization
    head=None
    last=None
    
    with open(input_path, 'rb') as f:
        ## Find boundaries
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_processes,
            split_special_token=split_special_token
        )
        
        ## Pretokenize
        for b_i in range(1, len(boundaries)):
            start = boundaries[b_i-1]
            end = boundaries[b_i]
                
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            parts = split_pat.split(chunk)

            for part in parts:
                if part in special_tokens:
                    # hit a boundary — strip it out *and* force a break
                    if last:
                        last.is_next_connected = False
                    continue

                # otherwise it's plain text: pretokenize it
                new_head, new_last = pretokenize(part, start_pos=start)

                if new_head is None:
                    continue

                if head is None:
                    head = new_head
                elif last is not None:
                    # connect across chunks *only* within a segment
                    last.next = new_head
                    new_head.prev = last
                    last.is_next_connected = False

                last = new_last
            
            
            # Remove Special Tokens
            # for x in special_tokens:
            #     chunk = chunk.replace(x, '')
            # # chunk = chunk.strip()
            
            # new_head, new_last = pretokenize(chunk, start_pos=start)
            
            # if head is None:
            #     head = new_head
            
            # if not last is None:
            #     last.next=new_head
            #     if new_head:
            #         new_head.prev=last
                
            #     # combined = (last.val + new_head.val).decode("utf-8", errors="ignore")
            #     # if full_token_re.fullmatch(combined):
            #     #     print("SPLIT BY CHUNK", combined)
            #     #     last.is_next_connected = True
            #     # else:
            #     #     last.is_next_connected = False
            #     last.is_next_connected=False
            
            # last = new_last
        
        
    # 3. Count, Record Positions
    pair_positions = defaultdict(set)
    node = head
    while node and node.next:
        # print(node.val, node.is_next_connected)
        if not node.is_next_connected:
            node=node.next
            continue
        
        pair_positions[
            (node.val, node.next.val)
        ].add(node)
        node = node.next
    
    pair_counts = {pair: len(nodes) for pair, nodes in pair_positions.items()}
    
    # 4. Merge
    num_merges = vocab_size-cur_vocab_size
    # print("CUR: {}, NUM MERGES {}".format(cur_vocab_size, num_merges))
    merges: List[Tuple[bytes, bytes]] = []
    
    remaining_merges = num_merges
    while remaining_merges:
    # for merge_i in range(num_merges):
        # Break Ties - preferring the lexicographically greater pair.
        max_count_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair[0],#.decode('utf-8', errors='ignore'),
                pair[1]#.decode('utf-8', errors='ignore')
            )
        )
            
        # Add to merges
        merges.append(max_count_pair)
        remaining_merges-=1
        
        # Add new vocab
        merged_val = b''.join(max_count_pair)
        vocab[cur_vocab_size]=merged_val
        cur_vocab_size+=1
        
        # print("MERGE {} {}".format(merge_i, merged_val))
        max_count_pair_positions = list(pair_positions[max_count_pair])
        max_count_pair_positions.sort(key=lambda x: x.span)
        
        for node_a in max_count_pair_positions:
            # Re-validate if still merge-able
            if (
                node_a.next is None
                or node_a.val!=max_count_pair[0]
                or not node_a.is_next_connected 
                or node_a.next.val!=max_count_pair[1]
            ):
                continue
            if not node_a in pair_positions[max_count_pair]:
                # print("HI")
                continue
            
            node_b = node_a.next
            
            # 1. Merge Node
            new_node = TokenNode(merged_val)
            new_node.prev=node_a.prev
            new_node.next=node_b.next
            new_node.is_next_connected=node_b.is_next_connected
            ## new span
            if node_a.span and node_b.span:
                new_span = (node_a.span[0], node_b.span[1])
                new_node.span=new_span
            
            # 2. Update Left
            if node_a.prev:
                if node_a.prev.is_next_connected:
                    # Remove previous
                    prev_pair = (node_a.prev.val, node_a.val)
                    pair_counts[prev_pair]-=1
                    pair_positions[prev_pair].discard(node_a.prev)
                    
                    # Add new merged version
                    new_pair = (node_a.prev.val, merged_val)
                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + 1
                    pair_positions[new_pair].add(node_a.prev)

                node_a.prev.next=new_node
            
            # 3. Update Right
            if node_b.next:
                if node_b.is_next_connected:
                    # Remove previous
                    prev_pair = (node_b.val, node_b.next.val)
                    pair_counts[prev_pair]-=1
                    pair_positions[prev_pair].discard(node_b)

                    # Add new merged version
                    new_pair = (merged_val, node_b.next.val)
                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + 1
                    pair_positions[new_pair].add(new_node)

                node_b.next.prev=new_node
            
            node_a.val=None
            node_b.val=None
            # del node_a
            # del node_b
        
        del pair_counts[max_count_pair]
        del pair_positions[max_count_pair]
    return vocab, merges

if __name__=='__main__':
    pass
