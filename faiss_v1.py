from langchain_community.document_loaders import JSONLoader
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
json_file_path = 'clean_data.json'  # Your input JSON file name

# Define the metadata extraction function
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["url"] = record.get("url")
    return metadata

# Initialize JSONLoader
loader = JSONLoader(
    file_path=json_file_path,
    jq_schema=".[]",  # This targets the entire JSON object
    content_key="content",
    metadata_func=metadata_func
)

# Load documents
docs = loader.load()

# Load or create vector store and retriever
faiss_index_path = "faiss_index"

if os.path.exists(faiss_index_path):
    # Load saved embeddings with dangerous deserialization allowed
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    st.success("Embeddings loaded from saved FAISS index.")
else:
    # Generate embeddings and save them
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    
    # Save the embeddings for future use
    vectorstore.save_local(faiss_index_path)
    st.success("Embeddings created and saved.")

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Question-answering chain setup
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you don't know."
    "\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
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
import time
# Handle new chat input
if user_input := st.chat_input("Ask a question from the ICIEC website?"):
    with st.chat_message("user"):
        st.markdown(user_input)

    # Print the user input (question) in the terminal
    print("User Input:", user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    import time

    # Inside your assistant response section, replace this part with streaming logic
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()  # Placeholder for "thinking" message
        message_placeholder = st.empty()  # Placeholder for streaming message
        full_response = ""

        # Display a "thinking" message while retrieving documents
        thinking_placeholder.markdown("🤔 Thinking...")

        # Retrieve the documents based on the input prompt
        retrieved_docs = retriever.invoke(user_input)

        # Once retrieval is done, replace the "thinking" message

        # Print the retrieved documents' content (context) being passed to the LLM
        print("Retrieved Context for LLM:")
        # for doc in retrieved_docs:
        #     print(doc.page_content)  # Print the paragraph/content sent to the model

        # Now invoke the LLM with the prompt and context
        response = rag_chain.invoke({"input": user_input})

        thinking_placeholder.empty()  # Remove the "thinking" message
        # Get the answer from the response
        full_response = response["answer"]

        # Break down the response into chunks for streaming effect
        chunks = full_response.split(" ")  # Split by words (or sentences)

        # Display chunks one by one to simulate streaming
        displayed_response = ""
        for chunk in chunks:
            displayed_response += chunk + " "
            message_placeholder.markdown(displayed_response + "▌")  # Show typing indicator
            time.sleep(0.05)  # Adjust delay for streaming effect

        # Gather the URLs from the retrieved documents
        retrieved_urls = set(doc.metadata.get("url") for doc in retrieved_docs if doc.metadata.get("url"))

        # If URLs exist, append them to the final response after the streaming
        if retrieved_urls:
            source_urls = "\n\nSources: " + ", ".join(retrieved_urls)
            full_response += source_urls
            displayed_response += source_urls

        # Display the final response without the typing indicator
        message_placeholder.markdown(displayed_response)

    # Save the full response in the session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
