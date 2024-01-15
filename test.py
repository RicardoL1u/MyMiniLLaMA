from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
model = LlamaForCausalLM.from_pretrained("MODELS/llama-7b-hf").to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("MODELS/llama-7b-hf")

# test generation by "hi"
input_str = "hi"
input_ids = tokenizer(input_str, return_tensors="pt")["input_ids"].to(model.device)
generated_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=True, top_p=0.9, top_k=0)
generated_str = tokenizer.decode(generated_ids[0])
print(generated_str)