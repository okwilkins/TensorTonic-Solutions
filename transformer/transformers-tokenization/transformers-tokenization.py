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
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        print(f"texts: {texts}")

        words = []
        for text in texts:
            for word in text.split(" "):
                words.append(word)
        words = sorted(words)

        for word in words:
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)
        
        self.id_to_word = {
            value: key
            for key, value in self.word_to_id.items()
        }
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        if text == "":
            return []

        return [
            self.word_to_id.get(word, self.word_to_id[self.unk_token])
            for word in text.lower().split(" ")
        ]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for id in ids:
            if id in self.id_to_word:
                words.append(self.id_to_word[id])
            else:
                words.append(self.unk_token)

        return " ".join(words)
