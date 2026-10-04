import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
MANUAL_PATH = ROOT / "data" / "mg-zs-warning-messages.html"
INDEX_DIR = ROOT / ".chroma"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant for the MG ZS car manual.
Use only the context below to answer the question.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{question}"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_manual():
    loader = UnstructuredHTMLLoader(file_path=str(MANUAL_PATH))
    return loader.load()


def split_manual(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def build_models():
    api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )
    return llm, embeddings


def open_vectorstore(chunks, embeddings):
    vectorstore = Chroma(
        collection_name="mg_zs_warnings",
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
    )
    if not vectorstore.get()["ids"]:
        vectorstore.add_documents(chunks)
    return vectorstore


def build_chain():
    documents = load_manual()
    chunks = split_manual(documents)
    llm, embeddings = build_models()
    vectorstore = open_vectorstore(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
    )
