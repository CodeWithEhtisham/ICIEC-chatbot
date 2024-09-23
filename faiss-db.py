# Importing required libraries
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import bs4

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
else:
    st.success("API Key Loaded")

# Initialize LangChain and OpenAI client
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Web page loader
loader_multiple_pages = WebBaseLoader(
    web_paths=[
        "https://iciec.isdb.org/",
        "https://iciec.isdb.org/climate-change/",
        "https://iciec.isdb.org/impact/",
        "https://iciec.isdb.org/covid-19/",
        "https://iciec.isdb.org/iciec-food-security/",
        "https://iciec.isdb.org/who-we-are/"
    ],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(name=['h1', 'h2', 'h3', 'h4', 'h5', 'p', 'a', 'span', 'textarea'])
    ),
    show_progress=True
)
docs = loader_multiple_pages.load()

# Text splitting and vector store creation
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()

# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Add custom CSS to style the header like a navbar
st.markdown(
    """
    <style>
    .navbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        border-bottom: 1px solid gray;
        padding-top: 40px;
        padding-left: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        font-size: 54px;
        color: gray;
        z-index: 1000;
    }
    .stApp {
        margin-top: 60px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use the custom class for the navbar
st.markdown('<div class="navbar">Welcome to ICIEC Info Assistant</div>', unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Handle new chat input
if prompt := st.chat_input("Ask a question from the web pages?"):
    # Replace specific words with summarization
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Use LangChain RAG Chain for question-answering
        response = rag_chain.invoke({"input": prompt})
        full_response = response["answer"]
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
