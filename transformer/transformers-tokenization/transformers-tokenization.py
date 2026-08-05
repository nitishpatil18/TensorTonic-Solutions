import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }

        words = set()
        for text in texts:
            for word in text.lower().split():
                words.add(word)

        for word in sorted(words):
            self.word_to_id[word] = len(self.word_to_id)

        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(word, unk_id) for word in text.lower().split()]
        pass
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        return " ".join(self.id_to_word.get(i, self.unk_token) for i in ids)
        pass
