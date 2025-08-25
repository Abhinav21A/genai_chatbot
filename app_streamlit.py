import streamlit as st
from transformers import pipeline

# Load GPT-2 Medium (local, free)
chatbot = pipeline("text-generation", model="gpt2-medium")

# Streamlit UI
st.set_page_config(page_title="Generative AI Chatbot", page_icon="🤖")
st.title("🤖 Generative AI Chatbot with Memory")
st.write("Chat with a personalized AI assistant that remembers the conversation.")

# Session state for conversation memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User inputs
user_name = st.text_input("Enter your name:", "User")
user_query = st.text_area("Ask me anything:")

if st.button("Get Response"):
    if user_query.strip() != "":
        # Add user message to history
        st.session_state.chat_history.append(f"{user_name}: {user_query}")

        # Build stronger conversation context
        conversation = "\n".join(st.session_state.chat_history)
        prompt = (
            f"The following is a friendly and concise conversation between {user_name} and an AI assistant. "
            f"The assistant always gives short, clear, and relevant answers.\n\n{conversation}\nAI:"
        )

        with st.spinner("Generating response..."):
            response = chatbot(
                prompt,
                max_length=150,         # limit length
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.5, # reduce looping
                pad_token_id=50256
            )

            # Extract reply and clean it
            reply = response[0]["generated_text"].replace(prompt, "").strip()

            # Only keep first sentence/short reply to avoid rambling
            if "." in reply:
                reply = reply.split(".")[0] + "."

            # Add AI reply to history
            st.session_state.chat_history.append(f"AI: {reply}")

        st.success(reply)
    else:
        st.warning("Please enter a query to get a response.")

# Display conversation history
st.subheader("📝 Conversation History")
for message in st.session_state.chat_history:
    if message.startswith("AI:"):
        st.markdown(f"**🤖 {message[3:]}**")
    else:
        st.markdown(f"**🧑 {message}**")
