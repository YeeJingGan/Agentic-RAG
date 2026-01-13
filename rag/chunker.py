import logging
from rich.logging import RichHandler
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rag")


class Chunker:
    def __init__(
            self,
            parent_chunk_size: int = 1000,
            parent_chunk_overlap: int = 100,
            child_chunk_size: int = 200,
            child_chunk_overlap: int = 20,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.parent_chunker = self._init_chunker_(chunk_size=self.parent_chunk_size,chunk_overlap=self.parent_chunk_overlap)
        self.child_chunker = self._init_chunker_(chunk_size=self.child_chunk_size,chunk_overlap=self.child_chunk_overlap)

    def _init_chunker_(self, chunk_size: int, chunk_overlap: int):
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="gpt2",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk(self, text:str, article_id:str) -> Tuple[List[Dict[str, Any]], List[Document]]:
        parent_docs = []
        child_docs = []

        parents = self.parent_chunker.create_documents([text])        
        for i, parent in enumerate(parents):
            parent_id = f"{article_id}-parent-{i}"
            parent_docs.append({
                "parent_id": parent_id,
                "article_id": article_id,
                "document": parent.page_content
            })

            children = self.child_chunker.create_documents([parent.page_content])
            for j, child in enumerate(children):
                child_id = f"{parent_id}-child-{j}"
                child_metadata = {
                    "id": child_id,
                    "parent_id": parent_id,
                    "article_id": article_id,
                }
                child_docs.append(Document(
                    page_content=child.page_content,
                    metadata=child_metadata
                ))

        log.info(f"CHUNKER PARENT: {len(parents)} chunks created for article_id: {article_id}")
        log.info(f"CHUNKER CHILD: {len(child_docs)} chunks created for article_id: {article_id}")

        return parent_docs, child_docs