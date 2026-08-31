from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import RAG_DOCUMENT_PATH, RAG_VECTORSTORE_PATH

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vectorstore():

    try:

        vectorstore = FAISS.load_local(
            RAG_VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    except Exception:

        loader = TextLoader(
            RAG_DOCUMENT_PATH,
            encoding="utf-8"
        )

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        vectorstore.save_local(
            RAG_VECTORSTORE_PATH
        )

        return vectorstore