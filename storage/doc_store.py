import sqlite3
import logging
from rich.logging import RichHandler
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rag")


class DocStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_tables()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_tables(self):
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    article_id TEXT PRIMARY KEY,
                    url TEXT,
                    title TEXT,
                    summary TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parents (
                    parent_id TEXT PRIMARY KEY,
                    article_id TEXT,
                    document TEXT,
                    FOREIGN KEY(article_id) REFERENCES articles(article_id)
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_parents_article
                ON parents(article_id)
                """
            )
        log.info("Database tables ready.")

    def add_articles(self, articles: List[Dict[str, Any]]):
        if not articles:
            return

        sql = """
            INSERT INTO articles (article_id, url, title, summary)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(article_id) DO UPDATE SET
                url = excluded.url,
                title = excluded.title,
                summary = excluded.summary
        """

        data = [(a["article_id"], a["url"], a["title"], a["summary"]) for a in articles]

        with self._get_conn() as conn:
            conn.executemany(sql, data)

        log.info(f"DOC STORE ADD_ARTICLES: {[(a['article_id'], a['title']) for a in articles]}")

    def add_parents(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return

        sql = """
            INSERT INTO parents (parent_id, article_id, document)
            VALUES (?, ?, ?)
            ON CONFLICT(parent_id) DO UPDATE SET
                article_id = excluded.article_id,
                document = excluded.document
        """

        data = [(c["parent_id"], c["article_id"], c["document"]) for c in chunks]

        with self._get_conn() as conn:
            conn.executemany(sql, data)

        log.info(f"DOC STORE ADD_PARENTS: {list({c['article_id'] for c in chunks})}")

    def get_parents(self, parent_ids: List[str]):
        if not parent_ids:
            return []

        placeholders = ",".join("?" for _ in parent_ids)
        with self._get_conn() as conn:
            cur = conn.execute(
                f"SELECT * FROM parents WHERE parent_id IN ({placeholders})",
                parent_ids,
            )
            rows = cur.fetchall()

        row_map = {row["parent_id"]: dict(row) for row in rows}

        return [row_map[pid] for pid in parent_ids if pid in row_map]

    def get_articles(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        if not article_ids:
            return []

        placeholders = ",".join("?" for _ in article_ids)
        with self._get_conn() as conn:
            cur = conn.execute(
                f"SELECT * FROM articles WHERE article_id IN ({placeholders})",
                article_ids,
            )
            rows = cur.fetchall()

        row_map = {row["article_id"]: dict(row) for row in rows}

        return [row_map[aid] for aid in article_ids if aid in row_map]

    def get_all_articles(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM articles")
            rows = cur.fetchall()

        article_titles = [row["title"] for row in rows]
        articles_list = [dict(row) for row in rows]

        log.info(f"DOC STORE GET_ALL_ARTICLES: {article_titles} articles")

        return articles_list

    def get_all_article_ids(self) -> List[str]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT article_id FROM articles")
            rows = cur.fetchall()
            article_ids = [row["article_id"] for row in rows]

        log.info(f"DOC STORE GET_ALL_ARTICLE_IDS: {article_ids}")

        return article_ids