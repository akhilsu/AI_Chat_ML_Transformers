import streamlit as st
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Set an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device_map="auto")
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def generate_response(message: str, history: list, temperature: float, max_new_tokens: int) -> str:
    """
    Generate a streaming response using the llama3-8b model.
    Args:
        message (str): The input message.
        history (list): The conversation history.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated response.
    """
    conversation = []
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators,
    )
    if temperature == 0:
        generate_kwargs['do_sample'] = False

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

# Streamlit interface
st.title("Meta Llama3 8B Chat")
st.markdown("This app demonstrates the instruction-tuned model Meta Llama3 8B.")

# Input elements
temperature = st.slider("Temperature", 0.0, 1.0, 0.95)
max_new_tokens = st.slider("Max new tokens", 128, 4096, 512)

# Chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("Ask me anything...")

if st.button("Send"):
    if user_input:
        st.session_state.history.append((user_input, ""))

        # Generate response
        with st.spinner("Generating response..."):
            response = ""
            for text in generate_response(user_input, st.session_state.history, temperature, max_new_tokens):
                response += text
                st.text_area("Response", response, height=200)
        
        # Append the generated response to history
        st.session_state.history[-1] = (user_input, response)

# Display chat history
for user, assistant in st.session_state.history:
    st.write(f"**User:** {user}")
    st.write(f"**Assistant:** {assistant}")
