import os
import logging
from dotenv import load_dotenv
from rich.logging import RichHandler
from typing import List, Dict, Any, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()
EMBEDDING_ID = os.getenv("EMBEDDING_MODEL_ID")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rag")


class Retriever:
    def __init__(self, vector_store, doc_store):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_ID, task_type="RETRIEVAL_QUERY"
        )
        self.vector_store = vector_store
        self.doc_store = doc_store

    def retrieve_children(
        self, query: str, k: int = 10, filter: Dict[str, Any] | None = None
    ) -> Tuple[List[str], List[str], List[str]]:
        docs = self.vector_store.search(
            query=query, k=k, filter=filter if filter else None
        )
        child_ids = [doc.metadata["id"] for doc in docs]
        parent_ids = list(dict.fromkeys(d.metadata["parent_id"] for d in docs))
        article_ids = list(dict.fromkeys(d.metadata["article_id"] for d in docs))

        return child_ids, parent_ids, article_ids

    def retrieve_parents(self, parent_ids: List[str]) -> List[str]:
        parents = self.doc_store.get_parents(parent_ids)
        documents = [row["document"] for row in parents]

        return documents

    def retrieve_articles(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        articles = self.doc_store.get_articles(article_ids)

        return articles

    def retrieve(
        self, query: str, k: int = 10, filter: Dict[str, Any] | None = None
    ) -> Tuple[
        List[str],  # child_ids
        List[str],  # parent_ids
        List[str],  # article_ids
        List[str],  # parent_docs
        List[Dict[str, Any]],  # articles
    ]:
        child_ids, parent_ids, article_ids = self.retrieve_children(
            query=query, k=k, filter=filter
        )
        parent_docs = self.retrieve_parents(parent_ids=parent_ids)
        articles = self.retrieve_articles(article_ids=article_ids)

        log.info(f"RETRIEVER: Children- {len(child_ids)}, Parents- {len(parent_docs)}, Articles- {len(articles)}")

        return child_ids, parent_ids, article_ids, parent_docs, articles
