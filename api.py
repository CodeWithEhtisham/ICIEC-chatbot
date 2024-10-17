from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specify which origins can access the API
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize LangChain and OpenAI client
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
faiss_index_path = "faiss_index"

# Load vector store
if os.path.exists(faiss_index_path):
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:
    raise Exception("FAISS index not found!")

# Create retriever and chains
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you don't know."
    "\n\n"
    "{context}"
)
prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query(request: QueryRequest):
    user_input = request.question
    response = rag_chain.invoke({"input": user_input})
    answer = response["answer"]
    print(answer)
    # Optionally, you could include source URLs here if needed
    return {"answer": answer}
