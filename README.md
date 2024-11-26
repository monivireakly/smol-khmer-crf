### Khmer Syllable Breaker

A Python tool for analyzing and generating Khmer syllable patterns using Conditional Random Fields (CRF).

## Overview

This tool analyzes Khmer text by:
- Classifying syllables into 6 distinct patterns (CV, CVC, CCV, CCVC, CCCV, CCCVC)
- Processing pre-segmented vocabulary files
- Generating new words based on learned syllable patterns
## Usage

```python
# Initialize analyzer
analyzer = KhmerSyllableAnalyzer()

# Process vocabulary file
analyzer.process_vocab('data/processed_vocab.txt')

# Generate single word
new_word = analyzer.generate_word(num_syllables=2)

# Generate dataset
dataset = analyzer.generate_dataset(num_words=1000)
```

## Input File Format

The vocabulary file (`processed_vocab.txt`) should contain Khmer words.

## Features

- Syllable pattern classification
- Word analysis and segmentation
- Statistics reporting
