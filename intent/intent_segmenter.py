import re

def split_into_segments(text: str) -> list[str]:
    """
    Splits a transcription into individual command segments using rule-based indicators.
    
    Indicators: 'and', 'then', 'also', 'plus', 'after that', ',', ';'
    
    Example:
        Input: "cancel my meeting and set alarm for 6"
        Output: ["cancel my meeting", "set alarm for 6"]
    """
    # Define indicators for splitting
    indicators = ["and", "then", "also", "plus", "after that"]
    
    # Create a regex pattern to split by indicators or punctuation
    # We use word boundaries for multi-word indicators to avoid partial matches
    pattern = r"|" .join([rf"\b{indicator}\b" for indicator in indicators])
    pattern += r"|[,;]"
    
    # Split the text based on the pattern
    segments = re.split(pattern, text, flags=re.IGNORECASE)
    
    # Clean up segments (strip whitespace and filter out empty strings)
    cleaned_segments = [seg.strip() for seg in segments if seg.strip()]
    
    return cleaned_segments

if __name__ == "__main__":
    # Test cases
    test_text = "cancel my meeting and set alarm for 6"
    print(f"Input: {test_text}")
    print(f"Output: {split_into_segments(test_text)}")
    
    test_text_2 = "send email to hnn; also check the weather"
    print(f"\nInput: {test_text_2}")
    print(f"Output: {split_into_segments(test_text_2)}")
