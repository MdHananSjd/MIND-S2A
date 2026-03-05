from intent.naive_bayes import predict_intent as nb_predict
from intent.transformer_classifier import predict_intents as tf_predict
from intent.intent_segmenter import split_into_segments

def detect_intents(text: str) -> list[dict]:
    """
    Connects all intent detection components with deduplication logic.
    
    Part 5: Segmented Multi-Intent Flow
    Logic:
    1. Detect whether the text contains multiple intents.
    2. If NOT multi-intent: call naive_bayes.predict_intent(text)
    3. If multi-intent: split into segments and call transformer_classifier.predict_intents(segment)
    4. Combine and deduplicate based on highest confidence.
    """
    
    # 1. Detect multi-intent indicators
    multi_intent_indicators = ["and", "then", "also", "plus", "after that", ",", ";"]
    text_lower = text.lower()
    
    is_multi_intent = any(indicator in text_lower for indicator in multi_intent_indicators)
    
    if not is_multi_intent:
        # 2. Single intent path (Naive Bayes)
        print("\n[Intent Router] Single intent detected. Routing to Naive Bayes.")
        return nb_predict(text)
    else:
        # 3. Multi-intent path (Transformer)
        print("\n[Intent Router] Multi-intent detected. Routing to Transformer Classifier.")
        segments = split_into_segments(text)
        
        all_predictions = []
        for segment in segments:
            # Get predictions for each segment
            segment_intents = tf_predict(segment)
            all_predictions.extend(segment_intents)
            
        # Part 4: Intent Deduplication
        # Keep the highest confidence score for each unique intent.
        deduplicated_map = {}
        for pred in all_predictions:
            intent_name = pred["intent"]
            confidence = pred["confidence"]
            
            # If same intent appears, keep the highest confidence
            if intent_name not in deduplicated_map or confidence > deduplicated_map[intent_name]:
                deduplicated_map[intent_name] = confidence
        
        # Part 6: Final Output Format
        # Convert dictionary back to required list format
        final_output = [
            {"intent": name, "confidence": conf}
            for name, conf in deduplicated_map.items()
        ]
        
        return final_output

if __name__ == "__main__":
    # Test cases for the improved router
    print("\n--- Improved Router Test: Single Intent ---")
    print(detect_intents("schedule meeting tomorrow"))
    
    print("\n--- Improved Router Test: Multi Intent Deduplication ---")
    # Simulate a case where multiple segments might hit the same intent
    multi_command = "cancel my meeting and also cancel the meeting for 5"
    print(f"Input: {multi_command}")
    print(detect_intents(multi_command))
