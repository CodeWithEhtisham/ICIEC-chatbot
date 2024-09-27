from langchain_community.document_loaders import JSONLoader
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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


# Commented out to suppress loaded data display
# for doc in docs:
#     st.write(doc.page_content)  # Display the content
#     st.write(doc.metadata)       # Display the metadata

# Text splitting and vector store creation
vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

retriever = vectorstore.as_retriever()

# Question-answering chain setup
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know."
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
# Handle new chat input
# Handle new chat input
if prompt := st.chat_input("Ask a question from the ICIEC website?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Print the user input (question) in the terminal
    print("User Input:", prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Use LangChain RAG Chain for question-answering
        # Retrieve the documents based on the input prompt
        retrieved_docs = retriever.get_relevant_documents(prompt)

        # Print the retrieved documents' content (context) being passed to the LLM
        print("Retrieved Context for LLM:")
        for doc in retrieved_docs:
            print(doc.page_content)  # Print the paragraph/content sent to the model

        # Now invoke the LLM with the prompt and context
        response = rag_chain.invoke({"input": prompt})

        # Get the answer from the response
        full_response = response["answer"]

        # Gather the URLs from the retrieved documents
        retrieved_urls = set(doc.metadata.get("url") for doc in retrieved_docs if doc.metadata.get("url"))

        # If URLs exist, append them to the response
        if retrieved_urls:
            source_urls = "\n\nSources: " + ", ".join(retrieved_urls)
            full_response += source_urls

        # Print the assistant's response to the terminal
        print("Assistant Response:", full_response)

        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
