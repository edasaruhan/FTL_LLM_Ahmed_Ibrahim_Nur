# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "bigscience/bloom-560m"  # or use "gpt2", "bigscience/bloom", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, max_length=100):
    """Generate text based on a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("Welcome to the SDG Text Generation App!")
    prompt = input("Enter your prompt related to SDGs: ")
    generated_text = generate_text(prompt)
    print("\nGenerated Text:\n", generated_text)

if __name__ == "__main__":
    main()
