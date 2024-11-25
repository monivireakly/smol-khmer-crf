from typing import List, Dict
import re
from pathlib import Path
import random

class KhmerSyllableAnalyzer:
    def __init__(self):
        self.patterns = {
            'CV': [],    # ក, ខ, គ
            'CVC': [],   # កក, ខក, គក
            'CCV': [],   # ក្រ, ខ្យ, គ្វ
            'CCVC': [],  # ក្រក, ខ្យក, គ្វក
            'CCCV': [],  # ស្ត្រ, ស្ក្រ
            'CCCVC': []  # ស្រុក
        }
        
        # Regex patterns for classification
        self.consonants = r'[កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអ]'
        self.vowels = r'[឴឵ាិីឹឺុូួើឿៀេែៃោៅំះៈ]'
        self.subscripts = r'្'
        
    def classify_syllable(self, syllable: str) -> str:
        """Classify a syllable into one of the six patterns"""
        # Remove any existing boundaries
        syllable = syllable.replace('|', '')
        
        # Count components
        c_count = len(re.findall(self.consonants, syllable))
        v_count = len(re.findall(self.vowels, syllable))
        s_count = len(re.findall(self.subscripts, syllable))
        
        #print(f"Analyzing syllable: {syllable}")
        #print(f"Consonants: {c_count}, Vowels: {v_count}, Subscripts: {s_count}")
        
        # Determine pattern
        pattern = 'Unknown'
        if s_count == 0:
            if c_count == 1:
                pattern = 'CV' if v_count > 0 else 'CVC'
        elif s_count == 1:
            pattern = 'CCV' if v_count > 0 else 'CCVC'
        elif s_count == 2:
            pattern = 'CCCV' if v_count > 0 else 'CCCVC'
        
        #print(f"Pattern: {pattern}")
        return pattern
    
    def analyze_word(self, word: str) -> List[Dict]:
        """Analyze syllables in a word"""
        syllables = word.split('|')
        result = []
        
        for syllable in syllables:
            pattern = self.classify_syllable(syllable)
            if pattern != 'Unknown':
                self.patterns[pattern].append(syllable)
                result.append({
                    'syllable': syllable,
                    'pattern': pattern
                })
        
        return result
    
    def process_vocab(self, file_path: str):
        """Process vocabulary file and categorize syllables"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if '|' in word:  # Only process pre-segmented words
                    self.analyze_word(word)
    
    def generate_word(self, num_syllables: int = None) -> str:
        """Generate a new word using random syllable patterns"""
        if num_syllables is None:
            num_syllables = random.randint(1, 4)
        
        syllables = []
        for _ in range(num_syllables):
            # Choose a random pattern
            pattern = random.choice(list(self.patterns.keys()))
            
            # If we have examples of this pattern, use one
            if self.patterns[pattern]:
                syllable = random.choice(self.patterns[pattern])
                syllables.append(syllable)
        
        return '|'.join(syllables)
    
    def generate_dataset(self, num_words: int = 1000) -> List[str]:
        """Generate a dataset of words with syllable boundaries"""
        dataset = []
        for _ in range(num_words):
            word = self.generate_word()
            if word:  # Only add non-empty words
                dataset.append(word)
        return dataset

def main():
    analyzer = KhmerSyllableAnalyzer()
    
    # Process existing vocab
    file_path = Path('processed_vocab.txt')
    
    if file_path.exists():
        print("Processing vocabulary...")
        
        # Read and preprocess the corpus
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            words = text.split()  # Split by whitespace
            
            # Process each word
            for word in words:
                # Basic syllable segmentation rules for Khmer
                syllables = []
                current = ""
                
                for i, char in enumerate(word):
                    current += char
                    
                    # Check for syllable boundaries
                    if i < len(word) - 1:
                        next_char = word[i + 1]
                        
                        # Rules for syllable breaks:
                        if (char in 'កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអ' and 
                            next_char not in '្ាិីឹឺុូួើឿៀេែៃោៅំះៈ'):
                            syllables.append(current)
                            current = ""
                            
                if current:
                    syllables.append(current)
                
                # Add to analyzer with boundaries
                segmented_word = '|'.join(syllables)
                analyzer.analyze_word(segmented_word)
        
        # Print statistics
        print("\nSyllable Pattern Statistics:")
        for pattern, syllables in analyzer.patterns.items():
            print(f"{pattern}: {len(set(syllables))} unique syllables")
            if syllables:  # Print some examples if available
                print("Examples:", ', '.join(list(set(syllables))[:5]))
        
        # Generate some example words
        print("\nGenerated Word Examples:")
        for _ in range(10):
            word = analyzer.generate_word()
            if word:
                print(f"Word: {word}")
                analysis = analyzer.analyze_word(word)
                patterns = [s['pattern'] for s in analysis]
                print(f"Patterns: {patterns}")
                print()

if __name__ == "__main__":
    main() 