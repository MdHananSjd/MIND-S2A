from perception.asr import transcribe
from intent.intent_router import detect_intents

def main():
    """
    Main entry point for the MIND-S2A modular AI cognitive agent.
    Perception -> Intent Detection
    """
    print("\n=== MIND-S2A Agent Started ===")
    
    # Layer 1: Perception (ASR)
    # This will prompt the user to record for 5 seconds by default
    text = transcribe()
    
    # Layer 2: Intent Detection (Routing + Classification)
    if text:
        intents = detect_intents(text)
        
        print("\n[Layer 2: Final Intent List]")
        print(intents)
    else:
        print("No transcription received. Please check your microphone.")

if __name__ == "__main__":
    main()
