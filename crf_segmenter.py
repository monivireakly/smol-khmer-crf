from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import pycrfsuite
from collections import Counter
from syllable_patterns import KhmerSyllableAnalyzer
import random

class KhmerCRFSegmenter:
    def __init__(self):
        self.tagger = pycrfsuite.Tagger()
        self.trainer = pycrfsuite.Trainer()
        
        # Enhanced parameters
        # self.trainer.set_params({
        #     'c1': 0.1,  # L1 regularization
        #     'c2': 0.05,  # L2 regularization
        #     'max_iterations': 100,
        #     'feature.possible_transitions': True,
        #     'feature.possible_states': True,
        #     'feature.minfreq': 3,
        #     'num_memories': 8,  # Increased from 6
        #     'epsilon': 1e-5,
        #     'period': 10,  # How often to compute the loss
        #     'delta': 1e-5,  # Convergence criterion
        #     'max_linesearch': 20,
        #     'linesearch': 'MoreThuente'
        # })
        self.trainer.set_params({
            'c1': 0.08,         # L1 regularization - slightly reduced to allow more features
            'c2': 0.02,         # L2 regularization - reduced to prevent over-smoothing
            'max_iterations': 200,  # Increased to allow more convergence time
            'feature.possible_transitions': True,
            'feature.possible_states': True,
            'feature.minfreq': 2,   # Reduced to capture more rare but potentially useful patterns
            'num_memories': 12,     # Increased for better optimization history
            'epsilon': 1e-6,        # Increased precision
            'period': 5,           # More frequent loss computation for better monitoring
            'delta': 1e-6,         # Tighter convergence criterion
            'max_linesearch': 25,   # More line search attempts
            'linesearch': 'MoreThuente',
            # 'feature.window_size': 5,  # Added: context window size
            #'feature.edge_feature': True,  # Added: special handling for word boundaries
            #'feature.include_bias': True   # Added: bias feature for better generalization
        })
        
        # Add callback for logging
        #self.trainer.on_iteration = self._iteration_callback
        self.current_epoch = 0
        self.max_log_epochs = 20
        
        # Character type mappings
        self.char_types = {
            'CONSONANT': 'កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអ',
            'VOWEL': 'ាិីឹឺុូួើឿៀេែៃោៅំះៈ',
            'SUBSCRIPT': '្',
            'NUMBER': '០១២៣៤៥៦៧៨៩',
            'DIACRITIC': '់ៈំះ'
        }
        
        # Syllable patterns from analysis
        self.patterns = {
            'CV': ['B-SYL'],
            'CVC': ['B-SYL', 'I-SYL'],
            'CCV': ['B-CC', 'I-SYL'],
            'CCVC': ['B-CC', 'I-SYL', 'E-WORD'],
            'CCCV': ['B-CC', 'I-CC', 'I-SYL'],
            'CCCVC': ['B-CC', 'I-CC', 'I-SYL', 'E-WORD']
        }
    
    # def _iteration_callback(self, context, log):
    #     """Callback for logging training progress"""
    #     self.current_epoch += 1
    #     if self.current_epoch <= self.max_log_epochs:
    #         print(f'Epoch {self.current_epoch:3d}'
    #               f' | Loss: {log.loss:.4f}'
    #               f' | Active Features: {log.num_active_features}'
    #               f' | Linesearch trials: {log.num_linesearch}'
    #               f' | Time: {log.time:.3f}s')
    
    def get_char_type(self, char: str) -> str:
        """Get character type feature"""
        for type_name, chars in self.char_types.items():
            if char in chars:
                return type_name
        return 'OTHER'
    
    def get_context_features(self, chars: List[str], pos: int) -> Dict[str, str]:
        """Enhanced feature extraction"""
        features = {
            'char': chars[pos],
            'char_type': self.get_char_type(chars[pos]),
            'is_first': pos == 0,
            'is_last': pos == len(chars) - 1,
            'char_position': str(pos),
            'word_length': str(len(chars))
        }
        
        # Window features
        window = 2
        for i in range(-window, window + 1):
            if 0 <= pos + i < len(chars):
                features[f'char[{i}]'] = chars[pos + i]
                features[f'type[{i}]'] = self.get_char_type(chars[pos + i])
        
        # Bigram features
        if pos > 0:
            features['bigram-1'] = chars[pos-1] + chars[pos]
        if pos < len(chars) - 1:
            features['bigram+1'] = chars[pos] + chars[pos+1]
        
        # Pattern features
        if pos > 1:
            features['pattern-2'] = ''.join(self.get_char_type(c)[0] for c in chars[pos-2:pos+1])
        if pos < len(chars) - 2:
            features['pattern+2'] = ''.join(self.get_char_type(c)[0] for c in chars[pos:pos+3])
        
        return features
    
    def word_to_features(self, word: str) -> List[Dict]:
        """Convert word to feature sequence"""
        chars = list(word)
        return [self.get_context_features(chars, i) for i in range(len(chars))]
    
    def get_labels(self, syllable_pattern: str) -> List[str]:
        """Get label sequence for a syllable pattern"""
        return self.patterns.get(syllable_pattern, ['O'] * len(syllable_pattern))
    
    def train(self, words_with_patterns: List[Tuple[str, List[str]]]):
        """Train CRF model on words with their syllable patterns"""
        print("Starting training with", len(words_with_patterns), "examples")
        
        for word, patterns in words_with_patterns:
            # Get features for each character
            features = self.word_to_features(word)
            labels = ['O'] * len(word)  # Initialize all as 'O'
            
            try:
                # Debug info
                # print(f"\nProcessing word: {word}")
                # print(f"Patterns: {patterns}")
                # print(f"Word length: {len(word)}")
                
                current_pos = 0
                for pattern in patterns:
                    pattern_length = 0
                    
                    # Calculate pattern length
                    if pattern == 'CV':
                        pattern_length = 1
                    elif pattern == 'CVC':
                        pattern_length = 2
                    elif pattern == 'CCV':
                        pattern_length = 2
                    elif pattern == 'CCVC':
                        pattern_length = 3
                    elif pattern == 'CCCV':
                        pattern_length = 3
                    elif pattern == 'CCCVC':
                        pattern_length = 4
                    
                    # Check if we have enough space for this pattern
                    if current_pos + pattern_length > len(word):
                        print(f"Warning: Pattern {pattern} doesn't fit at position {current_pos}")
                        continue
                    
                    # Assign labels based on pattern
                    if pattern == 'CV':
                        labels[current_pos] = 'B-SYL'
                    elif pattern == 'CVC':
                        labels[current_pos] = 'B-SYL'
                        labels[current_pos + 1] = 'I-SYL'
                    elif pattern == 'CCV':
                        labels[current_pos] = 'B-CC'
                        labels[current_pos + 1] = 'I-SYL'
                    elif pattern == 'CCVC':
                        labels[current_pos] = 'B-CC'
                        labels[current_pos + 1] = 'I-SYL'
                        labels[current_pos + 2] = 'E-WORD'
                    elif pattern == 'CCCV':
                        labels[current_pos] = 'B-CC'
                        labels[current_pos + 1] = 'I-CC'
                        labels[current_pos + 2] = 'I-SYL'
                    elif pattern == 'CCCVC':
                        labels[current_pos] = 'B-CC'
                        labels[current_pos + 1] = 'I-CC'
                        labels[current_pos + 2] = 'I-SYL'
                        labels[current_pos + 3] = 'E-WORD'
                    
                    current_pos += pattern_length
                
                # Debug info
                # print(f"Final labels: {labels}")
                # print(f"Features length: {len(features)}")
                # print(f"Labels length: {len(labels)}")
                
                # Only append if lengths match
                if len(features) == len(labels):
                    self.trainer.append(features, labels)
                else:
                    print(f"Warning: Length mismatch - Features: {len(features)}, Labels: {len(labels)}")
                    
            except Exception as e:
                print(f"Error processing word '{word}': {str(e)}")
                continue
    
    def segment(self, word: str) -> str:
        """Enhanced segmentation with confidence scores"""
        features = self.word_to_features(word)
        labels = self.tagger.tag(features)
        
        # Get marginal probabilities
        probs = [self.tagger.marginal(label, i) for i, label in enumerate(labels)]
        
        result = []
        current_syllable = ""
        
        for char, label, prob in zip(word, labels, probs):
            current_syllable += char
            if label in ['E-WORD', 'B-SYL'] and prob > 0.5:  # Only split if confidence > 50%
                result.append(current_syllable)
                current_syllable = ""
        
        if current_syllable:
            result.append(current_syllable)
            
        return '|'.join(result)

def main():
    analyzer = KhmerSyllableAnalyzer()
    file_path = Path('processed_vocab.txt')
    
    if not file_path.exists():
        raise FileNotFoundError("Processed vocab file not found!")
    
    print("Reading corpus...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        words = text.split()
    print(f"Found {len(words)} words in corpus")
    
    print("\nProcessing words...")
    analyzer_results = []
    for word in words:
        # First segment the word into syllables
        syllables = []
        current = ""
        
        for i, char in enumerate(word):
            current += char
            if i < len(word) - 1:
                next_char = word[i + 1]
                if (char in analyzer.consonants and 
                    next_char not in '្ាិីឹឺុូួើឿៀេែៃោៅំះៈ'):
                    syllables.append(current)
                    current = ""
        
        if current:
            syllables.append(current)
        
        # Now analyze each syllable
        word_patterns = []
        for syllable in syllables:
            pattern = analyzer.classify_syllable(syllable)
            if pattern != 'Unknown':
                word_patterns.append(pattern)
                # Add syllable to analyzer's pattern collection
                analyzer.patterns[pattern].append(syllable)
        
        if word_patterns:  # Only add if we got valid patterns
            analyzer_results.append((word, word_patterns))
            
        if len(analyzer_results) % 1000 == 0:
            print(f"Processed {len(analyzer_results)} words...")
    
    print(f"\nFound {len(analyzer_results)} valid words with patterns")
    
    # Print pattern statistics
    print("\nPattern Statistics:")
    for pattern, syllables in analyzer.patterns.items():
        unique_syllables = list(set(syllables))
        print(f"{pattern}: {len(unique_syllables)} unique syllables")
        if unique_syllables:
            print("Examples:", ', '.join(unique_syllables[:3]))
    
    if not analyzer_results:
        print("No valid training data found!")
        return
    
    # Initialize and train CRF
    segmenter = KhmerCRFSegmenter()
    
    # Filter valid examples before training
    valid_examples = []
    print("\nValidating examples...")
    for word, patterns in analyzer_results:
        # Calculate expected length based on patterns
        expected_length = sum(1 if p == 'CV' else
                            2 if p in ['CVC', 'CCV'] else
                            3 if p in ['CCVC', 'CCCV'] else
                            4 if p == 'CCCVC' else 0
                            for p in patterns)
        
        if expected_length == len(word):
            valid_examples.append((word, patterns))
    
    print(f"Found {len(valid_examples)} valid examples out of {len(analyzer_results)}")
    
    # Train with valid examples only
    if valid_examples:
        print("\nTraining CRF model...")
        print("Logging first 20 epochs:")
        print("-" * 70)
        print("Epoch |   Loss   | Active Features | Linesearch | Time (s)")
        print("-" * 70)
        
        segmenter.train(valid_examples[:1000])  # Start with a smaller subset for testing
        
        print("-" * 70)
        print("Training continues... (logging stopped after 20 epochs)")
        
        # Save the model
        print("Saving model...")
        segmenter.trainer.train('khmer_syllable_crf.model')
        
        # Test the model
        print("\nTesting model...")
        segmenter.tagger.open('khmer_syllable_crf.model')
        
        test_words = ["ការប្រកួត", "កីឡាករ", "កម្ពុជា", "លីមុនីវិរះ", "កីឡាករកម្ពុជាបានប្រកួតប្រជែងយ៉ាងសកម្មនៅក្នុងការប្រកួតកីឡាស៊ីហ្គេមលើកទី៣២នៅប្រទេសកម្ពុជា"]
        for word in test_words:
            segmented = segmenter.segment(word)
            print(f"{word} -> {segmented}")
    else:
        print("No valid examples found for training!")

if __name__ == "__main__":
    main() 