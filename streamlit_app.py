import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Page Config 
st.set_page_config(
    page_title="Yashoda — Medical Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #f0f4f8;
}

/* Header */
.yashoda-header {
    background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 60%, #118ab2 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.yashoda-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    margin: 0;
    letter-spacing: -0.5px;
}
.yashoda-header p {
    font-size: 1rem;
    opacity: 0.85;
    margin: 0.3rem 0 0 0;
    font-weight: 300;
}

/* Chat message bubbles */
.user-bubble {
    background: #1b6ca8;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0 0.5rem 20%;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(27,108,168,0.2);
}
.assistant-bubble {
    background: white;
    color: #1a2e40;
    border-radius: 18px 18px 18px 4px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 20% 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-left: 3px solid #118ab2;
}
.bubble-label {
    font-size: 0.72rem;
    font-weight: 500;
    opacity: 0.6;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f4c75;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label {
    color: rgba(255,255,255,0.8) !important;
    font-size: 0.85rem;
}

/* Input area */
.stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #d0dde8;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    background: white;
    transition: border 0.2s;
}
.stTextArea textarea:focus {
    border-color: #1b6ca8;
    box-shadow: 0 0 0 3px rgba(27,108,168,0.1);
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #0f4c75, #1b6ca8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.8rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* Disclaimer card */
.disclaimer {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #5d4037;
    margin-top: 1rem;
}

/* Medicine info card */
.med-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.med-card h4 {
    color: #0f4c75;
    font-family: 'DM Serif Display', serif;
    margin: 0 0 0.5rem 0;
    font-size: 1rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    flex: 1;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-card .num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #1b6ca8;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #78909c;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# Model Setup
MODEL_NAME = "tinyllama"

template = """
You are Yashoda, an expert medical assistant specializing in medicines.
Here is relevant information about the medicines from the database:
{context}

Based on the above information, answer the following question clearly and helpfully.
If the information is not available in the context, say so politely.

Question: {question}
"""

@st.cache_resource(show_spinner=False)
def load_chain():
    prompt = ChatPromptTemplate.from_template(template)
    model  = OllamaLLM(model=MODEL_NAME)
    return prompt | model

chain = load_chain()

# Session State 
if "messages" not in st.session_state:
    st.session_state.messages = []

#  Sidebar
with st.sidebar:
    st.markdown("##  Yashoda")
    st.markdown("**Medical Assistant**")
    st.markdown("---")

    st.markdown("### About")
    st.markdown(
        "Yashoda uses **RAG** (Retrieval-Augmented Generation) "
        "to answer medicine-related questions from a curated dataset."
    )
    st.markdown("---")

    st.markdown("### How to use")
    st.markdown("1. Type your medicine question\n2. Click **Ask Yashoda**\n3. Get an informed answer")
    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("### Developer")
    st.markdown("Mr.Naman Kumar\n\nEmail:namankumar1170@gmail.com")    

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; opacity:0.7'>⚠️ For informational use only.<br>Not a substitute for medical advice.</div>",
        unsafe_allow_html=True
    )

# Header 
st.markdown("""
<div class="yashoda-header">
    <h1> Yashoda</h1>
    <p>Your intelligent medical assistant — powered by Ollama & LangChain</p>
</div>
""", unsafe_allow_html=True)

# Sample Questions 
st.markdown("** Try asking:**")
sample_cols = st.columns(3)
samples = [
    "What are the side effects of Augmentin?",
    "What is Azithral 500 used for?",
    "What are substitutes for Augmentin?",
]
for col, sample in zip(sample_cols, samples):
    with col:
        if st.button(sample, use_container_width=True):
            st.session_state["prefill"] = sample
            st.rerun()

st.markdown("---")

# Chat History 
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble"><div class="bubble-label">You</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble"><div class="bubble-label">Yashoda</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

#  Input
prefill = st.session_state.pop("prefill", "")

with st.container():
    question = st.text_area(
        "Ask about any medicine:",
        value=prefill,
        placeholder="e.g., What are the side effects of Augmentin?",
        height=100,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask = st.button(" Ask Yashoda", use_container_width=True)

# Prediction 
if ask and question.strip():
    st.session_state.messages.append({"role": "user", "content": question.strip()})

    with st.spinner("Wait! I'm thinking..."):
        try:
            context = retriever.invoke(question.strip())
            answer  = chain.invoke({"context": context, "question": question.strip()})
            answer  = str(answer).strip()
        except Exception as e:
            answer = f" Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

elif ask and not question.strip():
    st.warning("Please enter a question first.")

# Disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> Yashoda provides general information only and is <strong>not</strong> a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)