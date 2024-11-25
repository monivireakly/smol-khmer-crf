from crf_segmenter import KhmerCRFSegmenter
from pathlib import Path

def test_model():
    # Initialize segmenter
    segmenter = KhmerCRFSegmenter()
    
    # Load trained model
    model_path = 'khmer_syllable_crf.model'
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file {model_path} not found!")
    
    segmenter.tagger.open(model_path)
    
    # Test sentences (complex cases)
    test_sentences = [
        # Long sentence with mixed patterns
        "កីឡាករកម្ពុជាបានប្រកួតប្រជែងយ៉ាងសកម្មនៅក្នុងការប្រកួតកីឡាស៊ីហ្គេមលើកទី៣២នៅប្រទេសកម្ពុជា",
        
        # Complex words with subscripts
        "ព្រះរាជាណាចក្រកម្ពុជា",
        "ស្ថាប័នអប់រំវិទ្យាសាស្ត្រនិងបច្ចេកវិទ្យា",
        
        # Numbers and special characters
        "ថ្ងៃទី១៥ ខែមករា ឆ្នាំ២០២៤",
        
        # Mixed scripts and patterns
        "ការប្រកួតបាល់ទាត់World Cupឆ្នាំ២០២៦",
        
        # Long compound words
        "អគ្គស្នងការដ្ឋាននគរបាលជាតិ",
        "មន្ទីរពេទ្យបង្អែកខេត្តកំពង់ចាម",
        
        # Academic terms
        "សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ",
        "វិទ្យាស្ថានបច្ចេកវិទ្យាកម្ពុជា",
        
        # Technical terms
        "ប្រព័ន្ធប្រតិបត្តិការណ៍កុំព្យូទ័រ",
        "បណ្ដាញអ៊ីនធឺណិតល្បឿនលឿន"
    ]
    
    print("Testing CRF Model on Complex Sentences")
    print("-" * 80)
    
    for sentence in test_sentences:
        segmented = segmenter.segment(sentence)
        print(f"\nOriginal: {sentence}")
        print(f"Segmented: {segmented}")
        print(f"Syllable count: {len(segmented.split('|'))}")
    
    print("\nDone testing!")

if __name__ == "__main__":
    test_model() 