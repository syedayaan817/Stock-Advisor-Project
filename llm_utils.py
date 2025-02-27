import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ✅ Enable quantization for faster inference (8-bit mode)
quantization = BitsAndBytesConfig(load_in_8bit=True)

# ✅ Correct Model Selection (Use GPT-2 with CausalLM)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change to another LLM if needed

# ✅ Load Tokenizer & Model (Now using AutoModelForCausalLM)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",  # ✅ Apply 8-bit optimization
)
# ✅ Automatically use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ✅ Function to generate response using local LLM

def generate_local_llm_response(prompt):
    """Generate a response using the local LLM and remove unwanted text."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=150,  # ✅ Keep responses short
            max_new_tokens=150,
            do_sample=True,
            temperature=0.3,
            top_k=50,
            top_p=0.9
        )


    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # ✅ Remove instructional text by filtering relevant lines
    extracted_lines = []
    for line in response.split("\n"):
        if "Company Name" in line or "Start Date" in line or "End Date" in line:
            extracted_lines.append(line.strip())

    extracted_response = "\n".join(extracted_lines)
  # Debugging

    return extracted_response if extracted_response else response  # ✅ Return extracted info