import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from intent.intent_labels import INTENT_LABELS
from transformers import logging
logging.set_verbosity_error()

# Global model and tokenizer
_model = None
_tokenizer = None

# Part 2: Prediction Threshold Tuning
THRESHOLD = 0.75

def load_model():
    """
    Part 1: Global Transformer Model Loading.
    Loads the DistilBERT model and tokenizer only once.
    """
    global _model, _tokenizer
    
    # If the model is already loaded, return immediately
    if _model is not None and _tokenizer is not None:
        return

    model_name = "distilbert-base-uncased"
    num_labels = len(INTENT_LABELS)
    
    print(f"\n[Transformer] Loading {model_name}...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    _model.eval()
    print("[Transformer] Model loaded successfully.")

def predict_intents(text: str) -> list[dict]:
    """
    Predicts intents for a given text segment using prompt-guided input.
    
    Returns:
        A list of dictionaries containing 'intent' and 'confidence'.
        Format: [{"intent": "...", "confidence": float}]
    """
    # Ensure model is loaded
    load_model()
    
    # Part 3: Prompt-Guided Transformer Input
    # Improve semantic recognition by providing context to the model
    possible_intents_str = ", ".join(INTENT_LABELS)
    guided_input = f"User command: {text}. Possible intents: {possible_intents_str}."
    
    # 1. Tokenize guided input
    inputs = _tokenizer(guided_input, return_tensors="pt", truncation=True, padding=True)
    
    # 2. Run forward pass
    with torch.no_grad():
        outputs = _model(**inputs)
    
    # 3. Apply sigmoid activation
    probs = torch.sigmoid(outputs.logits).squeeze().tolist()
    
    # Handle single label edge case
    if isinstance(probs, float):
        probs = [probs]
    
    # 4. Apply THRESHOLD (0.75) and Map predictions
    results = []
    for i, confidence in enumerate(probs):
        if confidence >= THRESHOLD:
            results.append({
                "intent": INTENT_LABELS[i],
                "confidence": round(confidence, 4)
            })
            
    # Fallback: if no intent meets the threshold, return the highest scoring one
    if not results and probs:
        max_idx = probs.index(max(probs))
        results.append({
            "intent": INTENT_LABELS[max_idx],
            "confidence": round(probs[max_idx], 4)
        })
        
    return results

# Initialize model on module import
load_model()
