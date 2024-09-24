import json
from dotenv import load_dotenv
import bs4
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from bs4 import BeautifulSoup
import re
from langchain_openai import ChatOpenAI

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
print(env_path)
load_dotenv(dotenv_path=env_path)  # take environment variables from .env.
print(os.environ.get("OPENAI_API_KEY"))

# Debugging: Check if the API key is loaded properly
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")
else:
    print(f"API Key Loaded: {openai_api_key[:5]}********")  # Only print the first few chars for security

os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize OpenAI LLM and Embeddings
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}
import re
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader

# Custom extractor using BeautifulSoup to get <h1>, <h2>, <span>, etc.
def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml-xml")
    # Extract only specified tags: h1, h2, h3, h4, h5, span, p
    # extracted_text = "\n\n".join([element.get_text(strip=True) for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'span', 'p'])])
    # # Remove excessive line breaks and return cleaned text
    # return re.sub(r"\n\n+", "\n\n", extracted_text).strip()
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()
# Create the RecursiveUrlLoader with the custom extractor
loader = RecursiveUrlLoader(
    url="https://iciec.isdb.org/",  # Example URL
    max_depth=2,
    headers=headers,
    extractor=bs4_extractor  # Pass the custom extractor function here
)

# Load the documents using the loader
docs = loader.load()

# Print the first document's content to verify
print(len(docs))
print(docs[0].page_content[:20])  # Preview of the parsed text

print(f"Number of documents loaded: {len(docs)}")

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print("############################ Splitting completed ##########################")

# Create a Chroma vector store from the document chunks
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Create a retriever from the vector store
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

# Define the prompt for the LLM
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Combine the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Start a loop to ask questions
while True:
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    response = rag_chain.invoke({"input": question})
    print("\n\n")
    print(response["answer"])
    print("\n\n")
