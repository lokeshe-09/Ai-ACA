import streamlit as st
import os
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
import streamlit.components.v1 as components

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*$$$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
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
                if i < len(self.api_keys) - 1:
                    self._switch_api_key((self.current_api_key_index + 1) % len(self.api_keys))
                else:
                    return "I apologize, but I'm currently experiencing technical difficulties. Please try again later.", False

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
    
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');
    
    body {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
    }
    
    .stApp {
        
    }
    
    #MainMenu, footer, header {display: none;}
    
    .main-title {
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(to right, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
    }
    
    @keyframes title-glow {
        from {
            text-shadow: 0 0 5px #ff6b6b, 0 0 10px #ff6b6b, 0 0 15px #ff6b6b, 0 0 20px #ff6b6b;
        }
        to {
            text-shadow: 0 0 10px #4ecdc4, 0 0 20px #4ecdc4, 0 0 30px #4ecdc4, 0 0 40px #4ecdc4;
        }
    }
    
    .welcome-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .feature-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .username-input {
        background: rgba(255, 255, 255, 0.2);
        border: none;
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        color: #ffffff;
        width: 100%;
        max-width: 400px;
        margin: 2rem auto;
        display: block;
    }
    
    .username-input::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }
    
    .enter-button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        cursor: pointer;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        display: block;
        margin: 0 auto;
    }
    
    .enter-button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .footer {
        text-align: center;
        padding: 5px 0;
        font-size: 14px;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 80px;
    }
    
    .particle {
        position: absolute;
        background: white;
        border-radius: 50%;
        pointer-events: none;
        opacity: 0.5;
    }
    
    @keyframes float-up {
        0% {
            transform: translateY(0) rotate(0deg);
            opacity: 1;
        }
        100% {
            transform: translateY(-100vh) rotate(360deg);
            opacity: 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def welcome_screen():
    st.markdown('<h1 class="main-title">Welcome to AI ACA ✨</h1>', unsafe_allow_html=True)
    
    username = st.text_input(
        label="Username",  # Add a non-empty label
        key="welcome_input", 
        placeholder="Enter your AI persona (e.g., QuantumDreamer42)",
        help="Your unique identifier in the AI ACA universe",
        label_visibility="collapsed"  # Hide the label visually
    )
    
    enter_button = st.button("🚀 Launch Your AI Journey", key="enter_button")
    
    st.markdown("""
    <p style="font-size: 1.2rem; text-align: center; margin-bottom: 2rem;">
        Embark on an extraordinary journey through the realms of artificial intelligence! 🚀
        AI ACA is your gateway to a world where imagination meets innovation.
    </p>
    
    <div class="feature-grid">
        <div class="feature-item">
            <i class="fas fa-robot feature-icon" style="color: #ff6b6b;"></i>
            <h3>AI Chat 💬</h3>
            <p>Engage in mind-bending conversations with our advanced AI. Unlock knowledge and spark creativity!</p>
        </div>
        <div class="feature-item">
            <i class="fas fa-image feature-icon" style="color: #4ecdc4;"></i>
            <h3>Image Generation 🎨</h3>
            <p>Transform your wildest ideas into stunning visuals. Watch your imagination come to life!</p>
        </div>
        <div class="feature-item">
            <i class="fas fa-globe feature-icon" style="color: #45aaf2;"></i>
            <h3>Website Analysis 🔍</h3>
            <p>Uncover hidden insights from any website. Let AI be your digital detective!</p>
        </div>
    </div>
    
    <p style="font-size: 1.2rem; text-align: center; margin-top: 2rem;">
        Ready to dive into the future? Create your unique AI persona and let the adventure begin! 🌟
    </p>
    """, unsafe_allow_html=True)
    
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add floating particles
    st.markdown("""
    <script>
    function createParticle() {
        const particle = document.createElement('div');
        particle.className = 'particle';
        document.body.appendChild(particle);
        
        const size = Math.random() * 5 + 5;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        
        const startX = Math.random() * window.innerWidth;
        particle.style.left = `${startX}px`;
        particle.style.bottom = '-20px';
        
        const duration = Math.random() * 3 + 2;
        particle.style.animation = `float-up ${duration}s linear infinite`;
        
        setTimeout(() => {
            particle.remove();
        }, duration * 1000);
    }
    
    setInterval(createParticle, 200);
    </script>
    """,
    unsafe_allow_html=True
    )
    
    return username, enter_button

def chat_interface():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "image":
                st.image(message["content"], caption="Generated Image", use_column_width=True)

    question = st.chat_input("Enter your question or image prompt:")

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "type": "text", "content": question})

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner(get_random_processing_message()):
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
                        response_placeholder.markdown('I apologize, but I\'m currently experiencing difficulties generating an image. Please try again later.')

def website_analysis_interface():
    url = st.text_input("Enter website URL:")
    website_question = st.text_area("Enter your question about the website:", height=100, key="website_question_input")
    if st.button("Analyze"):
        if url and website_question:
            response_placeholder = st.empty()
            with st.spinner(get_random_processing_message()):
                analysis, success = st.session_state.llama_chain.analyze_website(url, website_question, response_placeholder)
                if success:
                    st.markdown('<div class="response-area">', unsafe_allow_html=True)
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

def get_random_processing_message():
    messages = [
        "Brewing some AI magic... ✨",
        "Consulting the digital oracle... 🔮",
        "Crunching numbers at light speed... 💡",
        "Decoding the matrix... 🧠",
        "Summoning digital wisdom... 📚",
        "Channeling the power of AI... ⚡",
        "Weaving a tapestry of knowledge... 🕸️",
        "Diving into the data ocean... 🌊",
        "Igniting the neural networks... 🔥",
        "Embarking on a digital quest... 🗺️"
    ]
    return random.choice(messages)

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

    if 'username' not in st.session_state:
        username, enter_button = welcome_screen()
        if username and enter_button:
            logging.info(f"User {username} has entered the platform.")
            st.session_state.username = username
            st.rerun()
        return

    st.markdown(f'<h1 class="main-title">Welcome, {st.session_state.username}! 🎉</h1>', unsafe_allow_html=True)

    # Main content area
    if st.session_state.current_interface == "chat":
        st.markdown('<h2 class="section-title">Interact with AI ACA</h2>', unsafe_allow_html=True)
        chat_interface()
    else:
        st.markdown('<h2 class="section-title">Analyze Website</h2>', unsafe_allow_html=True)
        website_analysis_interface()

    # Footer with swap buttons
    st.markdown("""
    <div class='footer'>
    Developed by <a href='https://aicraftalchemy.github.io'>Ai Craft Alchemy</a><br>
    Connect with us: <a href='tel:+917661081043'>+91 7661081043</a>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    # Only show the swap chat/image button when in chat interface
    if st.session_state.current_interface == "chat":
        with col1:
            if st.session_state.current_mode == "chat":
                mode_label = "AI Chat 🤖"
                swap_label = "🔄 Swap to Image Generator 🖼️"
            else:
                mode_label = "Image Generator 🖼️"
                swap_label = "🔄 Swap to AI Chat 🤖"

            st.markdown(f'<p style="text-align: left ; font-weight: bold;">{mode_label}</p>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <style>
                .button1 {{
                    margin-top: 30px;  /* Move the button 20px down */
                }}
            </style>
        """, unsafe_allow_html=True)
            
            if st.button(swap_label, key="swap_mode", help="Switch between chat and image generation", args=('button1')):
                st.session_state.current_mode = "image" if st.session_state.current_mode == "chat" else "chat"
                st.rerun()

    with col2:            
        button_label = "🔄 Switch to Web Analyzer" if st.session_state.current_interface == "chat" else "🔄 Switch to Chat with AI"
        st.markdown(f"""
        <style>
            .button2 {{
                position: relative;  /* Use relative positioning to adjust */
                margin-top: 30px;  /* Move down */
                margin-left: 30px; /* Move right or left */
                margin-right: 50px; /* Adjust the right margin */
                padding: 10px; /* Adjust padding inside the button */
            }}
        </style>
    """, unsafe_allow_html=True)
        
        if st.button(button_label, key="swap_interface", help="Switch between chat and website analysis", args=('button2')):
            st.session_state.current_interface = "website" if st.session_state.current_interface == "chat" else "chat"
            st.rerun()

if __name__ == "__main__":
    create_streamlit_app()
