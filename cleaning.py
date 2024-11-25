import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import Counter
from typing import List, Dict
import json
from rich.console import Console
from rich.table import Table

# Khmer character classes
KHMER_CHARS = {
    'consonants': list('កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអ'),
    'subscripts': list('្ក្ខ្គ្ឃ្ង្ច្ឆ្ជ្ឈ្ញ្ដ្ឋ្ឌ្ឍ្ណ្ត្ថ្ទ្ធ្ន្ប្ផ្ព្ភ្ម្យ្រ្ល្វ្ឝ្ឞ្ស្ហ្ឡ្អ'),
    'vowels': list('ាិីឹឺុូួើឿៀេែៃោៅ'),
    'special_signs': list('់ះៈ៎ំាិី'),
    'independent_vowels': list('ឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ'),
    'numbers': list('០១២៣៤៥៦៧៨៩'),
    'diacritics': list('៉៊់័៌៍៎៏័៑្់៌៍'),
}

SYLLABLE_PATTERNS = {
    'C': 'CONSONANT',
    'V': 'VOWEL', 
    'S': 'SUBSCRIPT',
    'D': 'DIACRITIC'
}

class CharacterStats:
    def __init__(self):
        self.char_freq = {}
        self.bigram_freq = {}
        
    def update(self, words):
        for word in words:
            chars = list(word)
            for c in chars:
                self.char_freq[c] = self.char_freq.get(c, 0) + 1
            for c1, c2 in zip(chars, chars[1:]):
                bigram = (c1, c2)
                self.bigram_freq[bigram] = self.bigram_freq.get(bigram, 0) + 1
                
    def get_features(self, char_sequence, pos):
        char = char_sequence[pos]
        features = {
            'char_freq': self.char_freq.get(char, 0),
            'bigram_freq': 0
        }
        if pos < len(char_sequence)-1:
            next_char = char_sequence[pos+1]
            bigram = (char, next_char)
            features['bigram_freq'] = self.bigram_freq.get(bigram, 0)
        return features

class KhmerDataset(Dataset):
    def __init__(self, examples, include_metadata=False):
        self.examples = examples
        self.include_metadata = include_metadata
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        item = {
            'word': example['word'],
            'char_sequence': example['char_sequence'],
            'features': example['features'],
            'boundaries': torch.tensor(example['boundaries'], dtype=torch.float),
            'length': len(example['char_sequence'])
        }
        return item

def get_char_type(char):
    for char_type, chars in KHMER_CHARS.items():
        if char in chars:
            return char_type.upper()
    return 'OTHER'

def get_char_features(char_sequence, pos):
    window_size = 2
    features = {
        'char': char_sequence[pos],
        'char_type': get_char_type(char_sequence[pos]),
        'position': pos / len(char_sequence),
        'prev_chars': [char_sequence[max(0, pos-i)] for i in range(1, window_size+1)],
        'prev_types': [get_char_type(char_sequence[max(0, pos-i)]) for i in range(1, window_size+1)],
        'next_chars': [char_sequence[min(len(char_sequence)-1, pos+i)] for i in range(1, window_size+1)],
        'next_types': [get_char_type(char_sequence[min(len(char_sequence)-1, pos+i)]) for i in range(1, window_size+1)]
    }
    return features

def match_syllable_pattern(char_types):
    patterns = [
        ['C', 'V'],           # CV
        ['C', 'V', 'C'],      # CVC
        ['C', 'S', 'C', 'V'], # CSCV
        ['C', 'S', 'C']       # CSC
    ]
    
    for pattern in patterns:
        if len(char_types) != len(pattern):
            continue
        if all(SYLLABLE_PATTERNS[p] == t for p, t in zip(pattern, char_types)):
            return True
    return False

def is_syllable_boundary(chars, pos):
    if pos >= len(chars) - 1:
        return True
    
    curr, next_char = chars[pos], chars[pos + 1]
    curr_type = get_char_type(curr)
    next_type = get_char_type(next_char)
    
    # Core rules
    if curr_type == 'SUBSCRIPT' or next_type == 'SUBSCRIPT':
        return False
    if curr_type == 'CONSONANT' and (next_type == 'VOWEL' or next_type == 'DIACRITIC'):
        return False
    if curr_type == 'VOWEL' and next_type == 'CONSONANT':
        return True
    if curr_type == 'VOWEL' and next_type == 'VOWEL':
        return False
    
    return False

def process_word(word, char_stats=None):
    chars = list(word)
    features = []
    boundaries = []
    
    for i in range(len(chars)):
        char_features = get_char_features(chars, i)
        
        # Add syllable pattern features
        window = 3
        char_types = [get_char_type(c) for c in chars[max(0,i-window):i+window+1]]
        char_features['matches_pattern'] = match_syllable_pattern(char_types)
        
        if char_stats:
            char_features.update(char_stats.get_features(chars, i))
            
        features.append(char_features)
        boundaries.append(1 if is_syllable_boundary(chars, i) else 0)
        
    return {
        'word': word,
        'char_sequence': chars,
        'features': features,
        'boundaries': boundaries
    }

def load_vocab(path='data/processed_vocab.txt'):
    file_path = Path(path)
    vocab = []
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found!")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        vocab = f.read().split()
    return vocab[:10000]

class KhmerDataAnalyzer:
    def __init__(self, words: List[str]):
        self.words = words
        self.console = Console()
        self.stats = self._compute_stats()
        
    def _compute_stats(self) -> Dict:
        char_counter = Counter()
        type_counter = Counter()
        lengths = []
        syllable_counts = []
        
        for word in self.words:
            chars = list(word)
            lengths.append(len(chars))
            
            # Count characters and their types
            for char in chars:
                char_counter[char] += 1
                type_counter[get_char_type(char)] += 1
            
            # Count syllables
            boundaries = [i for i in range(len(chars)) if is_syllable_boundary(chars, i)]
            syllable_counts.append(len(boundaries) + 1)
        
        return {
            'total_words': len(self.words),
            'unique_chars': len(char_counter),
            'char_frequencies': dict(char_counter.most_common(10)),
            'type_frequencies': dict(type_counter),
            'avg_length': sum(lengths) / len(lengths),
            'avg_syllables': sum(syllable_counts) / len(syllable_counts),
            'max_length': max(lengths),
            'min_length': min(lengths)
        }
    
    def print_summary(self):
        """Print formatted summary of the dataset"""
        self.console.print("\n[bold blue]Khmer Dataset Summary[/bold blue]")
        self.console.print(f"Total words: {self.stats['total_words']:,}")
        self.console.print(f"Unique characters: {self.stats['unique_chars']}")
        self.console.print(f"Average word length: {self.stats['avg_length']:.2f}")
        self.console.print(f"Average syllables per word: {self.stats['avg_syllables']:.2f}")
        
        # Character type distribution table
        type_table = Table(title="Character Type Distribution")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="magenta")
        type_table.add_column("Percentage", style="green")
        
        total_chars = sum(self.stats['type_frequencies'].values())
        for char_type, count in self.stats['type_frequencies'].items():
            percentage = (count / total_chars) * 100
            type_table.add_row(
                char_type,
                str(count),
                f"{percentage:.1f}%"
            )
        
        self.console.print(type_table)
        
        # Top characters table
        char_table = Table(title="Top 10 Most Common Characters")
        char_table.add_column("Character", style="cyan")
        char_table.add_column("Count", style="magenta")
        char_table.add_column("Type", style="green")
        
        for char, count in self.stats['char_frequencies'].items():
            char_table.add_row(
                char,
                str(count),
                get_char_type(char)
            )
            
        self.console.print(char_table)

def main():
    # Load vocabulary
    vocab = load_vocab()
    
    # Initialize analyzer
    analyzer = KhmerDataAnalyzer(vocab)
    
    # Print analysis
    analyzer.print_summary()
    
    # Process a sample word to show features
    sample_word = vocab[0]
    processed = process_word(sample_word)
    
    print("\nSample Word Analysis:")
    print(f"Word: {sample_word}")
    print("Features for first character:")
    print(json.dumps(processed['features'][0], indent=2, ensure_ascii=False))
    print(f"Boundaries: {processed['boundaries']}")

if __name__ == "__main__":
    main()

