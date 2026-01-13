import re
import logging
import wikipedia
from rich.logging import RichHandler
from typing import List, Dict, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rag")

class WikipediaProcessor:
    def __init__(self, vector_store, doc_store, chunker):
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.chunker = chunker

    def _fetch_article(self, title: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        page_title = wikipedia.search(query=title, results=1, suggestion=True)[0]
        page = wikipedia.page(title=page_title, auto_suggest=False)

        if page:
            cleaned = re.sub(r"(?m)^\s*(=+)\s*[^=\n]+\s*\1\s*$", "", page.content, flags=re.DOTALL)

            article = {
                "article_id": f"article-{str(page.pageid)}",
                "url": page.url,
                "title": page.title,
                "summary": page.summary,
            }

            text = {
                "text": cleaned,
                "article_id": article["article_id"],
            }

            return text, article
        
    def process(self, titles: List[str]) -> List[Dict[str, Any]]:
        article_metadatas = []
        article_texts = []

        db_article_ids = self.doc_store.get_all_article_ids() # Existing article IDs in the database

        for title in titles:
            try:
                text, article_metadata = self._fetch_article(title)
                if text["article_id"] not in db_article_ids:
                    article_metadatas.append(article_metadata)
                    article_texts.append(text)
                    log.info(f"Fetched article '{title}' successfully.")
                else:
                    log.info(f"Article '{title}' already exists in the database. Skipping.")
            except Exception as e:
                log.error(f"Error fetching article '{title}': {e}")

        self.doc_store.add_articles(articles=article_metadatas)
        
        for text in article_texts:
            parent_docs, child_docs = self.chunker.chunk(
                text=text["text"],
                article_id=text["article_id"]
            )
            self.doc_store.add_parents(chunks=parent_docs)
            self.vector_store.add(documents=child_docs)
        
            