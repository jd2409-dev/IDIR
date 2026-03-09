import torch
import torch.nn.functional as F
from idir_model import IDIR

try:
    from transformers import GPT2Tokenizer
except ImportError:
    print("Install transformers: pip install transformers")
    raise

# Configuration matching training
vocab_size = 50257
hidden_dim = 512
num_experts = 4
expert_hidden_dim = 2048
reasoning_steps = 3
k_features = 32
reasoning_hidden_dim = 2560
max_iterations = 20
tolerance = 1e-3

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", model_max_length=1024)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Load model
model = IDIR(
    vocab_size=vocab_size,
    hidden_dim=hidden_dim,
    num_experts=num_experts,
    expert_hidden_dim=expert_hidden_dim,
    reasoning_steps=reasoning_steps,
    k_features=k_features,
    reasoning_hidden_dim=reasoning_hidden_dim,
    max_iterations=max_iterations,
    tolerance=tolerance,
)

# Load trained weights (assuming saved after training)
model.load_state_dict(torch.load("idir_model.pt"))
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_text(prompt, max_len=50, greedy=False, temperature=0.7):
    model.eval()
    with torch.no_grad():
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]

        for _ in range(max_len):
            logits, _ = model(input_ids)
            next_token_logits = logits[:, -1, :]
            if greedy:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # Return only the generated part
    prompt_text = tokenizer.decode(input_ids[0][:prompt_len], skip_special_tokens=True)
    generated_only = generated_text[len(prompt_text) :].strip()
    return generated_only


# Example interaction
if __name__ == "__main__":
    prompt = "The future of AI is"
    generated = generate_text(prompt)
    print(f"Prompt: {prompt}")
    safe_output = generated.encode("ascii", errors="replace").decode("ascii")
    print(f"Generated: {safe_output}")
