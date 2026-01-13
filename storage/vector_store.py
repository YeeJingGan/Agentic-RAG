import os
import faiss
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv
from rich.logging import RichHandler
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore


load_dotenv()
EMBEDDING_ID = os.getenv("EMBEDDING_MODEL_ID")
ROOT_DIR = Path(__file__).resolve().parent.parent
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rag")


class VectorStore:
    def __init__(self):
        self.embedding_function = GoogleGenerativeAIEmbeddings(model=EMBEDDING_ID, task_type="RETRIEVAL_DOCUMENT")
        self.dim = len(self.embedding_function.embed_query("hello world"))
        self.index = faiss.IndexFlatIP(self.dim)
        self.docstore = InMemoryDocstore()
        self.vector_store_path = ROOT_DIR / "faiss_index"
        self.store = self._setup_vector_store()

    def _setup_vector_store(self):
        if self.vector_store_path.exists():
            log.info("VECTOR STORE SETUP: Existing")
            store = FAISS.load_local(
                folder_path=self.vector_store_path,
                embeddings=self.embedding_function,
                allow_dangerous_deserialization=True
            )
        else:
            log.info("VECTOR STORE SETUP: New")
            store = FAISS(
                embedding_function=self.embedding_function,
                index=self.index,
                docstore=self.docstore,
                index_to_docstore_id={}
            )
            store.save_local(self.vector_store_path)

        return store


    def add(self, documents: List[Document]) -> int:
        existing_ids = set(self.docstore._dict.keys())

        new_docs = []
        for doc in documents:
            if doc.metadata["id"] not in existing_ids:
                new_docs.append(doc)
        
        if not new_docs:
            return 

        self.store.add_documents(
            documents=new_docs,
            ids=[str(uuid.uuid4()) for _ in new_docs]
        )
        self.store.save_local(self.vector_store_path)

        return len(new_docs)

    def search(self, query: str, k: int = 10, filter: Dict[str, Any] | None = None) -> List[Document]:
        results = self.store.similarity_search(
            query=query,
            k=k,
            filter=filter if filter else None
        )

        return results
    
    def delete(self, ids: List[str]) -> int:
        self.store.delete(ids=ids)
        self.store.save_local(self.vector_store_path)

        return len(ids)
