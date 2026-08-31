from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from rag_service import create_vectorstore

from config import OPENAI_MODEL


vectorstore = create_vectorstore()

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0
)


@tool
def student_rag(question: str) -> str:
    """
    Answer questions about the student using information
    retrieved from the student knowledge base.
    """
    print("[TOOL] student_rag")
    print(f"[INPUT] {question}")

    documents = vectorstore.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
Answer the user's question using only the provided context.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say that the information is not available.
"""

    response = llm.invoke(prompt)

    return response.content