### Khmer Syllable Breaker

A Python tool for analyzing and generating Khmer syllable patterns using Conditional Random Fields (CRF).

## Overview

This tool analyzes Khmer text by:
- Classifying syllables into 6 distinct patterns (CV, CVC, CCV, CCVC, CCCV, CCCVC)
- Processing pre-segmented vocabulary files
- Generating new words based on learned syllable patterns

## Syllable Patterns

The analyzer recognizes these patterns:
- `CV`: Single consonant + vowel (ក, ខ, គ)
- `CVC`: Single consonant + vowel + consonant (កក, ខក, គក) 
- `CCV`: Two consonants + vowel (ក្រ, ខ្យ, គ្វ)
- `CCVC`: Two consonants + vowel + consonant (ក្រក, ខ្យក, គ្វក)
- `CCCV`: Three consonants + vowel (ស្ត្រ, ស្ក្រ)
- `CCCVC`: Three consonants + vowel + consonant (ស្ត្រក, ស្ក្រក)

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

The vocabulary file (`processed_vocab.txt`) should contain Khmer words with syllable boundaries marked by `|` character:

ក្រ|ក
ស្ត្រ|ក
គ្វ|ក

## Features

- Syllable pattern classification
- Word analysis and segmentation
- Random word generation
- Dataset generation
- Statistics reporting
