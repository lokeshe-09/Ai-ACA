import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import re
import requests
import io
from PIL import Image
import logging
from collections import deque
from datetime import datetime, timedelta
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class LlamaAIChain:
    def __init__(self):
        self.api_keys = [
            os.getenv("GROQ_API_KEY"),
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2")
        ]
        self.current_api_key_index = 0
        self.llm = self._create_llm()
        self.memory = ConversationBufferMemory(return_messages=True)
        self.token_limits = {key: 20000 for key in self.api_keys}  # 20k tokens per minute limit
        self.token_usage = {key: deque(maxlen=60) for key in self.api_keys}  # Track usage for last 60 seconds

    def _create_llm(self):
        return ChatGroq(
            temperature=0.7,
            groq_api_key=self.api_keys[self.current_api_key_index],
            model_name="llama-3.1-70b-versatile"
        )

    def _switch_api_key(self, index):
        self.current_api_key_index = index
        self.llm = self._create_llm()
        logging.info(f"Switched to API key {self.current_api_key_index}")

    def _update_token_usage(self, tokens):
        current_key = self.api_keys[self.current_api_key_index]
        now = datetime.now()
        self.token_usage[current_key].append((now, tokens))

        # Remove entries older than 60 seconds
        while self.token_usage[current_key] and now - self.token_usage[current_key][0][0] > timedelta(seconds=60):
            self.token_usage[current_key].popleft()

    def _get_available_tokens(self, api_key):
        now = datetime.now()
        # Remove entries older than 60 seconds
        while self.token_usage[api_key] and now - self.token_usage[api_key][0][0] > timedelta(seconds=60):
            self.token_usage[api_key].popleft()

        used_tokens = sum(tokens for _, tokens in self.token_usage[api_key])
        return self.token_limits[api_key] - used_tokens

    def _get_best_api_key(self, required_tokens):
        available_tokens = [self._get_available_tokens(key) for key in self.api_keys]
        best_key_index = max(range(len(available_tokens)), key=lambda i: available_tokens[i])

        if available_tokens[best_key_index] >= required_tokens:
            return best_key_index
        else:
            return None

    def _wait_for_token_availability(self, required_tokens):
        while True:
            best_key_index = self._get_best_api_key(required_tokens)
            if best_key_index is not None:
                self._switch_api_key(best_key_index)
                return

            sleep_time = 1  # Wait for 1 second before checking again
            logging.info(f"Waiting for {sleep_time} seconds for token availability")
            time.sleep(sleep_time)

    def _try_operation(self, operation_func, placeholder, required_tokens):
        for i in range(len(self.api_keys)):
            self._wait_for_token_availability(required_tokens)
            logging.info(f"Attempting operation with API key {self.current_api_key_index}")
            try:
                result = operation_func()
                self._update_token_usage(required_tokens)
                logging.info(f"Operation successful with API key {self.current_api_key_index}")
                return result, True
            except Exception as e:
                logging.error(f"Error with API key {self.current_api_key_index}: {str(e)}")
                self._switch_api_key((self.current_api_key_index + 1) % len(self.api_keys))
                placeholder.markdown(get_random_waiting_message())
        
        return "I apologize, but I'm currently experiencing difficulties. Please try again later.", False

    def ask_question(self, question, placeholder):
        prompt = PromptTemplate(
            input_variables=["history", "question"],
            template="Chat History:\n{history}\nHuman: {question}\n\nAI: Let me think about that and provide a helpful response."
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)

        def operation():
            return chain.run(question=question)

        # Estimate token usage (this is a rough estimate, adjust as needed)
        estimated_tokens = len(question.split()) + 100  # Add some buffer for the prompt

        return self._try_operation(operation, placeholder, estimated_tokens)

    def analyze_website(self, url, question, placeholder):
        loader = WebBaseLoader([url])
        data = clean_text(loader.load()[0].page_content)

        prompt = PromptTemplate(
            input_variables=["website_content", "question"],
            template="""
            Analyze the following website content and answer the user's question:

            Website Content:
            {website_content}

            User's Question:
            {question}

            Provide a detailed and informative answer based on the website content:
            """
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        def operation():
            return chain.run(website_content=data, question=question)

        # Estimate token usage (this is a rough estimate, adjust as needed)
        estimated_tokens = len(data.split()) + len(question.split()) + 200  # Add some buffer for the prompt

        return self._try_operation(operation, placeholder, estimated_tokens)

def set_page_config():
    st.set_page_config(page_title="AI ACA", page_icon="✨", layout="wide", menu_items=None)
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: Arial, sans-serif;
    }
    .main-title {
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-title {
        color: #2c3e50;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .response-area {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .stTextInput>div>div>textarea {
        min-height: 100px;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
      width: 20%;
    }
    .chat-message .avatar img {
      max-width: 78px;
      max-height: 78px;
      border-radius: 50%;
      object-fit: cover;
    }
    .chat-message .message {
      width: 80%;
      padding: 0 1.5rem;
      color: #fff;
    }
    .footer {
        text-align: center;
        padding: 20px 0;
        font-size: 14px;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 40px;
    }
    .swap-button {
        margin-top: 10px;
    }
    .input-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .mode-label {
        font-weight: bold;
        margin-right: 10px;
    }
    .mode-indicator {
        font-weight: bold;
        margin-left: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_random_waiting_message():
    messages = [
        "Hang tight! I'm working on your request...",
        "Just a moment while I process that...",
        "Your patience is appreciated. I'm crunching the data...",
        "Almost there! This is an interesting query...",
        "I'm putting my AI brain to work on your request...",
        "Analyzing and formulating a response for you...",
        "This is a great question! Let me think about it...",
        "Processing... Your request is in good hands!",
        "I'm diving deep into my knowledge base for this one...",
        "Exciting query! I'm working on a thoughtful response..."
    ]
    return random.choice(messages)

def chat_interface():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "image":
                st.image(message["content"], caption="Generated Image", use_column_width=True)

    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    question = st.chat_input("Enter your question or image prompt:")
    st.markdown('</div>', unsafe_allow_html=True)

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "type": "text", "content": question})

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown(get_random_waiting_message())
            with st.spinner("Processing..."):
                if st.session_state.current_mode == "chat":
                    answer, success = st.session_state.llama_chain.ask_question(question, response_placeholder)
                    if success:
                        response_placeholder.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "type": "text", "content": answer})
                    else:
                        response_placeholder.markdown(answer)
                else:
                    image = generate_image(question)
                    if image:
                        response_placeholder.image(image, caption="Generated Image", use_column_width=True)
                        st.session_state.chat_history.append({"role": "assistant", "type": "image", "content": image})
                    else:
                        response_placeholder.markdown("I'm having trouble generating an image. Please try again later.")

def website_analysis_interface():
    url = st.text_input("Enter website URL:")
    website_question = st.text_area("Enter your question about the website:", height=100, key="website_question_input")
    if st.button("Analyze"):
        if url and website_question:
            response_placeholder = st.empty()
            response_placeholder.markdown(get_random_waiting_message())
            with st.spinner("Analyzing website..."):
                analysis, success = st.session_state.llama_chain.analyze_website(url, website_question, response_placeholder)
                if success:
                    st.markdown("<div class='response-area'>", unsafe_allow_html=True)
                    st.write("Analysis:", analysis)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(analysis)
        else:
            st.warning("Please enter both a URL and a question.")

def generate_image(prompt):
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content

    try:
        image_bytes = query({
            "inputs": prompt,
        })
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logging.error(f"Error in generate_image: {str(e)}")
        return None

def create_streamlit_app():
    set_page_config()

    if 'llama_chain' not in st.session_state:
        st.session_state.llama_chain = LlamaAIChain()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'current_interface' not in st.session_state:
        st.session_state.current_interface = "chat"

    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "chat"

    st.markdown("<h1 class='main-title'>✨ AI Platform by Ai Craft Alchemy</h1>", unsafe_allow_html=True)

    # Main content area
    if st.session_state.current_interface == "chat":
        st.markdown("<h2 class='section-title'>Interact with AI ACA</h2>", unsafe_allow_html=True)
        chat_interface()
    else:
        st.markdown("<h2 class='section-title'>Analyze Website</h2>", unsafe_allow_html=True)
        website_analysis_interface()

    # Footer with swap buttons
    st.markdown("""
    <div class='footer'>
    Developed by <a href='https://aicraftalchemy.github.io'>Ai Craft Alchemy</a><br>
    Connect with us: <a href='tel:+917661081043'>+91 7661081043</a>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Only show the swap chat/image button when in chat interface
    if st.session_state.current_interface == "chat":
        with col1:
            if st.session_state.current_mode == "chat":
                mode_label = "AI Chat 🤖"
                swap_label = "🔄Swap for Image Generator 🖼️"
            else:
                mode_label = "Image Generator 🖼️"
                swap_label = "🔄Swap to Interact with AI ACA 🤖"

            st.markdown(f'<span class="mode-indicator">{mode_label}</span>', unsafe_allow_html=True)
            if st.button(swap_label, key="swap_mode", help="Switch between chat and image generation"):
                st.session_state.current_mode = "image" if st.session_state.current_mode == "chat" else "chat"
                st.rerun()

    with col2:
        button_label = "Switch to Web Analyzer" if st.session_state.current_interface == "chat" else "Switch to Chat with AI"
        if st.button(button_label, key="swap_interface"):
            st.session_state.current_interface = "website" if st.session_state.current_interface == "chat" else "chat"
            st.rerun()

if __name__ == "__main__":
    create_streamlit_app()
