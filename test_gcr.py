import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Specify the model name
model_id = "rmanluo/GCR-Meta-Llama-3.1-8B-Instruct"

# 2. Load the tokenizer
# Using trust_remote_code=True might be necessary if the model has custom code,
# but it's good practice to be aware of the implications.
# For widely used models like Llama 3 variants, this is often standard.
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("If this is related to custom code, you might need to add 'trust_remote_code=True'.")
    exit()

# 3. Load the model
# This model is 8 billion parameters, so it requires significant RAM.
# - If you have a CUDA-enabled GPU with enough VRAM (e.g., >16GB for 8B in float16),
#   this will run much faster.
# - `torch_dtype=torch.bfloat16` (or torch.float16) reduces memory and can speed up inference.
# - `device_map="auto"` (requires `accelerate`) will try to optimally use available hardware (GPUs, CPU).
# For CPU-only, it will be slow and require substantial RAM.
try:
    print(f"Loading model: {model_id}. This may take a while and consume significant RAM/VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for reduced memory and potentially faster inference
        device_map="auto",          # Automatically maps model to available devices (GPU if available, else CPU)
                                    # Remove device_map or set to "cpu" if you only want to force CPU and have issues.
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure you have enough RAM/VRAM. For CPU, you might need >32GB RAM for an 8B model.")
    print("If you have a GPU, ensure CUDA is set up correctly.")
    exit()

# 4. Define the conversation content
user_question = 'Which city is the person who wrote the lyrics for "Long Live" from?'
prompt_instruction = "Please generate 5 reasoning paths to answer the question:"
full_user_message = f"{prompt_instruction} {user_question}"

# 5. Construct the chat prompt using the tokenizer's chat template
# Llama 3.1 Instruct models use a specific format for conversations.
# The `apply_chat_template` method handles this correctly.
messages = [
    {"role": "user", "content": full_user_message},
]

# The tokenizer will add the special tokens like <|begin_of_text|>, <|start_header_id|>, etc.
# `add_generation_prompt=True` ensures the template ends with the cue for the assistant to start generating.
try:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device) # Move tokenized input to the same device as the model
except Exception as e:
    print(f"Error applying chat template: {e}")
    exit()

# 6. Generate the response
# Adjust generation parameters as needed.
print("\nGenerating response...")
try:
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,  # Maximum number of new tokens to generate
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,      # Whether to use sampling; set to False for deterministic output (if model supports)
        temperature=0.6,     # Controls randomness. Lower is more deterministic.
        top_p=0.9,           # Nucleus sampling.
        pad_token_id=tokenizer.eos_token_id # Often set to eos_token_id for open-ended generation
    )
    # The output includes the input prompt, so we need to decode only the newly generated tokens.
    response_ids = outputs[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    print("\nGenerated Reasoning Paths/Answer:")
    print(response_text)

except Exception as e:
    print(f"Error during generation: {e}")

print("\nScript finished.")