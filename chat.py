import sys
from pathlib import Path

import torch
from transformers import GPT2Tokenizer
import argparse

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from idir_model import IDIRModel

def main(args):
    """Main function to run the chatbot."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = IDIRModel(vocab_size=tokenizer.vocab_size, hidden_dim=512)
    
    if args.checkpoint_path:
        try:
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
            print("Model checkpoint loaded successfully.")
        except FileNotFoundError:
            print("Checkpoint not found. Starting with a randomly initialized model.")
    
    model.to(device)
    model.eval()

    print("\nIDIR Chatbot. Type 'quit' to exit.")
    print("="*40)

    while True:
        prompt = input("You: ")
        if prompt.lower() == "quit":
            break

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=args.max_length,
                num_beams=args.num_beams,
                temperature=args.temperature,
                top_k=args.top_k,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Bot: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the IDIR model.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/phase_4_epoch_0.pt", help="Path to the model checkpoint.")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length.")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    
    args = parser.parse_args()
    main(args)
