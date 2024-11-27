import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Retrieve the Hugging Face token from environment variables
HF_TOKEN = os.environ.get("GColab", None)

# Ensure the token is provided
if HF_TOKEN is None:
    st.error("Hugging Face token is not set. Please set the HF_TOKEN environment variable.")
else:
    # Load the tokenizer and model
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available

        # Streamlit app setup
        st.set_page_config(
            page_title="AI Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Sidebar
        st.sidebar.title("Chatbot Settings")
        max_length = st.sidebar.slider("Response Length", min_value=50, max_value=512, value=100)
        temperature = st.sidebar.slider("Temperature", min_value=0.5, max_value=1.5, value=0.95)
        st.sidebar.markdown("Adjust the response settings for the chatbot.")

        # Main page
        st.title("🤖 Chat with AI")
        st.markdown("Welcome to the interactive AI chatbot. Start a conversation by typing below:")

        # Chat container
        if "conversation" not in st.session_state:
            st.session_state.conversation = []

        def generate_response(user_input, history=[], max_new_tokens=512, temperature=0.95):
            # Prepare the conversation history
            conversation = [{"role": "user", "content": msg} for msg in history]
            conversation.append({"role": "user", "content": user_input})

            # Tokenize the input
            input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)

            # Generate response
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode and return the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("assistant")[1]

        def add_to_conversation(user_input, bot_response):
            st.session_state.conversation.append({"user": user_input, "bot": bot_response})

        # User input
        user_input = st.text_input("You:", "")
        if st.button("Send") and user_input:
            bot_response = generate_response(user_input, max_new_tokens=max_length, temperature=temperature)
            add_to_conversation(user_input, bot_response)
            user_input = ""  # Clear the input field

        # Display conversation
        for chat in st.session_state.conversation:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Bot:** {chat['bot']}")

        # Footer
        st.markdown("---")
        st.markdown("Developed by Akhil Sudhakaran")

    except Exception as e:
        st.error(f"An error occurred: {e}")
