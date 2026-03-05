import math
from collections import Counter, defaultdict


class IntentClassifier:
    """
    Manual Multinomial Naive Bayes implementation
    for single-intent classification in MIND-S2A.
    """

    def __init__(self):
        # ----------------------------
        # Training Dataset
        # ----------------------------
        self.training_data = {
            "set_alarm": [
                "set alarm at 6",
                "wake me up at 7",
                "alarm for tomorrow morning"
            ],
            "schedule_meeting": [
                "schedule meeting at 10",
                "book appointment tomorrow",
                "add meeting to calendar"
            ],
            "cancel_event": [
                "cancel my meeting",
                "delete the alarm",
                "remove event from calendar"
            ]
        }

        self.intents = list(self.training_data.keys())

        # Model storage
        self.vocabulary = set()
        self.word_counts = defaultdict(Counter)
        self.total_words_per_intent = defaultdict(int)
        self.priors = {}

        self._train()

    # ----------------------------
    # Tokenization
    # ----------------------------
    def _tokenize(self, text: str):
        return text.lower().strip().split()

    # ----------------------------
    # Training
    # ----------------------------
    def _train(self):
        total_documents = sum(len(docs) for docs in self.training_data.values())

        for intent, documents in self.training_data.items():
            # Prior probability P(intent)
            self.priors[intent] = len(documents) / total_documents

            for doc in documents:
                tokens = self._tokenize(doc)
                for token in tokens:
                    self.vocabulary.add(token)
                    self.word_counts[intent][token] += 1
                    self.total_words_per_intent[intent] += 1

        self.vocab_size = len(self.vocabulary)

    # ----------------------------
    # Likelihood with Laplace smoothing
    # ----------------------------
    def _get_likelihood(self, word: str, intent: str):
        """
        P(word | intent) with Laplace smoothing.

        Even unseen words receive a small non-zero probability
        due to +1 smoothing.
        """
        word_count = self.word_counts[intent].get(word, 0)
        total_words = self.total_words_per_intent[intent]

        return (word_count + 1) / (total_words + self.vocab_size)

    # ----------------------------
    # Classification
    # ----------------------------
    def classify(self, text: str, verbose: bool = True):
        tokens = self._tokenize(text)
        log_posteriors = {}

        if verbose:
            print("\n[Layer 2: Bayesian Intent Classification]")

        for intent in self.intents:
            log_score = math.log(self.priors[intent])

            if verbose:
                print(f"\nIntent: {intent}")
                print(f"  Prior: {self.priors[intent]:.4f} "
                      f"(log: {math.log(self.priors[intent]):.4f})")
                print("  Likelihood Contributions:")

            for word in tokens:
                likelihood = self._get_likelihood(word, intent)
                log_likelihood = math.log(likelihood)
                log_score += log_likelihood

                if verbose:
                    print(f"    - '{word}': "
                          f"P={likelihood:.6f}, log={log_likelihood:.4f}")

            log_posteriors[intent] = log_score

            if verbose:
                print(f"  Final Log Score: {log_score:.4f}")

        # -------- Log-Sum-Exp Normalization --------
        max_log = max(log_posteriors.values())

        exp_scores = {
            intent: math.exp(score - max_log)
            for intent, score in log_posteriors.items()
        }

        total_exp = sum(exp_scores.values())

        normalized_probs = {
            intent: score / total_exp
            for intent, score in exp_scores.items()
        }

        predicted_intent = max(normalized_probs, key=normalized_probs.get)
        confidence = normalized_probs[predicted_intent]

        if verbose:
            print("\nNormalized Probabilities:")
            for intent, prob in normalized_probs.items():
                print(f"  {intent}: {prob:.4f}")

            print(f"\nPredicted Intent: {predicted_intent}")
            print(f"Confidence: {confidence:.4f}")

        return {
            "intent": predicted_intent,
            "confidence": confidence,
            "intent_probabilities": normalized_probs
        }


def predict_intent(text: str) -> list[dict]:
    """
    Wrapper function to provide a standardized interface for Naive Bayes classification.
    
    Returns:
        List containing a single dictionary: [{"intent": "...", "confidence": float}]
    """
    classifier = IntentClassifier()
    result = classifier.classify(text, verbose=False)
    
    return [
        {
            "intent": result["intent"],
            "confidence": round(result["confidence"], 4)
        }
    ]


# ----------------------------
# Standalone Test
# ----------------------------
if __name__ == "__main__":
    # Test individual classification
    print(predict_intent("schedule meeting tomorrow"))
