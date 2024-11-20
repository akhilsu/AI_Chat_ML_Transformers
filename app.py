import streamlit as st
from transformers import pipeline

# Initialize the text generation pipeline with GPT-Neo
model_name = "EleutherAI/gpt-neo-125M"  # A smaller version of GPT-Neo for testing
text_generator = pipeline("text-generation", model=model_name)

# Streamlit app setup
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("Chatbot Settings")
max_length = st.sidebar.slider("Response Length", min_value=50, max_value=200, value=100)
st.sidebar.markdown("Adjust the response length for the chatbot.")

# Main page
st.title("🤖 Chat with AI")
st.markdown("Welcome to the interactive chatbot. Start a conversation by typing below:")

# Chat container
if "conversation" not in st.session_state:
    st.session_state.conversation = []

def generate_response(user_input):
    try:
        responses = text_generator(
            user_input,
            max_length=max_length,
            num_return_sequences=1,
            truncation=True,       # Ensure truncation is explicitly set
            padding="max_length"   # Use padding to ensure consistent input length
        )
        return responses[0]['generated_text']
    except ValueError as e:
        st.error(f"An error occurred: {e}")
        return "I'm sorry, I encountered an error processing your request."

def add_to_conversation(user_input, bot_response):
    st.session_state.conversation.append({"user": user_input, "bot": bot_response})

# User input
user_input = st.text_input("You:", "")
if st.button("Send") and user_input:
    bot_response = generate_response(user_input)
    add_to_conversation(user_input, bot_response)

# Display conversation
for chat in st.session_state.conversation:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")

# Footer
st.markdown("---")
st.markdown("Developed by Akhil Sudhakaran")
