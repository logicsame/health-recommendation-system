import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer from Hugging Face
model_id = "huggingface/llama-2-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

st.title("Food Nutrition Analysis with LLaMA")

user_input = st.text_area("Enter food label text:")

if st.button("Analyze"):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(response)
