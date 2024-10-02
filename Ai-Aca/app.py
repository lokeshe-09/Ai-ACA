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
        self.token_limits = {key: 20000 for key in self.api_keys}
        self.token_usage = {key: deque(maxlen=60) for key in self.api_keys}
        self.error_count = 0
        self.max_retries = 3

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
        
        while self.token_usage[current_key] and now - self.token_usage[current_key][0][0] > timedelta(seconds=60):
            self.token_usage[current_key].popleft()

    def _get_available_tokens(self, api_key):
        now = datetime.now()
        while self.token_usage[api_key] and now - self.token_usage[api_key][0][0] > timedelta(seconds=60):
            self.token_usage[api_key].popleft()
        
        used_tokens = sum(tokens for _, tokens in self.token_usage[api_key])
        return max(0, self.token_limits[api_key] - used_tokens)

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
            
            sleep_time = 1
            logging.info(f"Waiting for {sleep_time} seconds for token availability")
            time.sleep(sleep_time)

    def _try_operation(self, operation_func, placeholder, required_tokens):
        self.error_count = 0
        while self.error_count < self.max_retries:
            try:
                self._wait_for_token_availability(required_tokens)
                result = operation_func()
                self._update_token_usage(min(required_tokens, self._get_available_tokens(self.api_keys[self.current_api_key_index])))
                self.error_count = 0  # Reset error count on successful operation
                return result, True
            except Exception as e:
                self.error_count += 1
                logging.error(f"Attempt {self.error_count} failed: {str(e)}")
                
                if "Rate limit reached" in str(e):
                    wait_time = re.search(r"Please try again in (\d+\.\d+)s", str(e))
                    if wait_time:
                        wait_seconds = float(wait_time.group(1))
                        placeholder.markdown(f"""
                        🕒 I'm processing your request. Please wait for {wait_seconds:.2f} seconds...
                        
                        I'll automatically continue once the waiting period is over.
                        """)
                        time.sleep(wait_seconds + 1)  # Add 1 second buffer
                        continue
                
                if self.error_count < self.max_retries:
                    placeholder.markdown(f"""
                    🔄 I apologize for the delay. I'm experiencing a temporary issue, but I'm trying again...
                    
                    Attempt {self.error_count + 1} of {self.max_retries}
                    """)
                    time.sleep(2)  # Wait before retrying
                else:
                    error_message = f"""
                    📝 I apologize, but I'm having trouble processing your request at the moment.
                    
                    Please try refreshing the page and trying again in a few moments.
                    
                    Our team has been notified and is working to resolve any issues.
                    
                    Thank you for your patience! 🙏
                    """
                    return error_message, False
        
        return "An unexpected error occurred. Please refresh the page and try again.", False

    def ask_question(self, question, placeholder):
        prompt = PromptTemplate(
            input_variables=["history", "question"],
            template="Chat History:\n{history}\nHuman: {question}\n\nAI: Let me think about that and provide a helpful response."
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)
        
        def operation():
            return chain.run(question=question)
        
        estimated_tokens = len(question.split()) + 100
        
        return self._try_operation(operation, placeholder, estimated_tokens)

    def analyze_website(self, url, question, placeholder):
        try:
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
            
            estimated_tokens = len(data.split()) + len(question.split()) + 200
            
            return self._try_operation(operation, placeholder, estimated_tokens)
        except Exception as e:
            logging.error(f"Error loading website: {str(e)}")
            return """
            🌐 I apologize, but I couldn't access that website at the moment.
            
            This might be because:
            - The URL might be incorrect
            - The website might be temporarily unavailable
            - The website might be blocking automated access
            
            Would you mind:
            1. Double-checking the URL
            2. Trying again in a few moments
            3. Perhaps trying a different website
            
            Thank you for your understanding! 🙏
            """, False

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
            with st.spinner("Processing..."):
                if st.session_state.current_mode == "chat":
                    answer, success = st.session_state.llama_chain.ask_question(question, response_placeholder)
                    response_placeholder.markdown(answer)
                    if success:
                        st.session_state.chat_history.append({"role": "assistant", "type": "text", "content": answer})
                else:
                    image = generate_image(question)
                    if image:
                        response_placeholder.image(image, caption="Generated Image", use_column_width=True)
                        st.session_state.chat_history.append({"role": "assistant", "type": "image", "content": image})
                    else:
                        response_placeholder.markdown("""
                        🎨 I apologize, but I'm having trouble generating the image at the moment.
                        
                        Please try refreshing the page and trying again in a few moments.
                        
                        Our team has been notified and is working to resolve any issues.
                        
                        Thank you for your patience! 🙏
                        """)

def website_analysis_interface():
    url = st.text_input("Enter website URL:")
    website_question = st.text_area("Enter your question about the website:", height=100, key="website_question_input")
    if st.button("Analyze"):
        if url and website_question:
            response_placeholder = st.empty()
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

    if st.session_state.current_interface == "chat":
        st.markdown("<h2 class='section-title'>Interact with AI ACA</h2>", unsafe_allow_html=True)
        chat_interface()
    else:
        st.markdown("<h2 class='section-title'>Analyze Website</h2>", unsafe_allow_html=True)
        website_analysis_interface()

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
            mode_label = "AI Chat 🤖" if st.session_state.current_mode == "chat" else "Image Generator 🖼️"
            swap_label = "🔄Swap for Image Generator 🖼️" if st.session_state.current_mode == "chat" else "🔄Swap to Interact with AI ACA 🤖"
            
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
