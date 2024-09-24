import json
# from scrapegraphai.graphs import SmartScraperGraph
from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import ChatOpenAI
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



llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)

loader_multiple_pages = WebBaseLoader(
    web_paths=[
        "https://iciec.isdb.org/",
        # "https://iciec.isdb.org/climate-change/",
        # "https://iciec.isdb.org/impact/",
        # "https://iciec.isdb.org/covid-19/",
        # "https://iciec.isdb.org/iciec-food-security/",

        ],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            # class_=("post-content", "post-title", "post-header")
            name=['h1','h2','h3','h4','h5','h6','p','a','span','textarea','ol','li','strong','span','ul']
        )
    ),
    show_progress=True
    )
docs = loader_multiple_pages.load()
print(len(docs))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print("############################ spliting completed ##########################")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

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
while True:
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    response = rag_chain.invoke({"input": question})
    print("\n\n")
    print(response["answer"])
    print("\n\n")

