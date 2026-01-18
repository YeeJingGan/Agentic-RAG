import os
from pathlib import Path
from dotenv import load_dotenv
from rag.chunker import Chunker
from typing import List, Any
from rag.retriever import Retriever
from langgraph.runtime import Runtime
from storage.doc_store import DocStore
from storage.vector_store import VectorStore
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import before_model
from langchain.agents import create_agent, AgentState
from rag.wikipedia_processor import WikipediaProcessor
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy


# --------------------------------------------------------
# Configuration and Initialization
# --------------------------------------------------------
load_dotenv()
PROJECT_ROOT = Path().resolve()
MODEL_ID = os.environ["LLM_MODEL_ID"]

vector_store = VectorStore()
doc_store = DocStore(db_path=PROJECT_ROOT / "doc_store.db")
chunker = Chunker(
    parent_chunk_size=1000,
    parent_chunk_overlap=100,
    child_chunk_size=200,
    child_chunk_overlap=20,
)
retriever = Retriever(
    vector_store=vector_store,
    doc_store=doc_store,
)
wiki_processor = WikipediaProcessor(
    chunker=chunker,
    vector_store=vector_store,
    doc_store=doc_store,
)

all_time_state = {
    "query": None,
    "chat_history": [],
    "should_retrieve": True,
    "final_k": 5,
}

# --------------------------------------------------------
# Agent Definitions and Middleware
# --------------------------------------------------------


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Keep only the last few messages to fit context window.

    Reference:
        https://docs.langchain.com/oss/python/langchain/short-term-memory
    """
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-8:]
    new_messages = [first_msg] + recent_messages

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]}


def wikipedia_tool(titles: List[str]):
    existing_articles = doc_store.get_all_articles()
    existing_titles = {article["title"] for article in existing_articles}
    for title in titles:
        if title in existing_titles:
            titles.remove(title)
    wiki_processor.process(titles=titles)


def retrieve_tool(query: str, k: int) -> List[str]:
    _, _, _, parent_docs, articles = retriever.retrieve(query=query, k=k)

    return parent_docs


# --------------------------------------------------------
# Define Structured Output Models
# --------------------------------------------------------


class RewriteDecision(BaseModel):
    should_rewrite: bool
    rewritten_query: str | None
    reason: str


class KnowledgeUpdateDecision(BaseModel):
    should_update: bool
    new_titles: List[str] | None
    reason: str


class MultipleRetrievalDecision(BaseModel):
    should_retrieve: bool
    reason: str


# --------------------------------------------------------
# Model and Agent Creation
# ---------------------------------------------------------

llm = ChatGoogleGenerativeAI(model=MODEL_ID, temperature=0)

# Agent 1: Query Rewrite Agent
query_rewrite_agent = create_agent(
    model=llm,
    response_format=ToolStrategy(RewriteDecision),
    system_prompt=(
        "You are a Query Optimization Agent.\n"
        "Your task is to make the user's query clear and explicit for an LLM to answer.\n"
        "Comprehend the user's query AND the chat history.\n"
        "If the query is not clear or relies on context from previous conversation:\n"
        "- set should_rewrite = true\n"
        "- include relevant information from chat history to make the query self-contained\n"
        "- provide rewritten_query\n"
        "- provide reason for rewriting the query\n"
        "Otherwise:\n"
        "- set should_rewrite = false\n"
        "- rewritten_query must be null\n"
        "- provide reason for not rewriting the query\n"
        "Rules:\n"
        "1. Only include information from chat history that is necessary for clarity.\n"
        "2. Do not assume facts not mentioned in chat history.\n"
        "3. Keep the rewritten query concise but self-contained."
    ),
)

# Agent 2: Knowledge Base Update Agent
knowledge_update_agent = create_agent(
    model=llm,
    response_format=ToolStrategy(KnowledgeUpdateDecision),
    system_prompt=(
        "You are a Knowledge Base Update Agent.\n"
        "Your task is to decide if new Wikipedia articles are needed to answer the user query.\n"
        "Input: user query + existing articles (title + summary).\n"
        "If the existing articles do NOT cover the query:\n"
        "- set should_update = true\n"
        "- provide new_titles = list of relevant Wikipedia article titles NOT in existing articles\n"
        "- provide reason for update\n"
        "Otherwise:\n"
        "- set should_update = false\n"
        "- new_titles = null\n"
        "- provide reason for no update\n"
        "Rules:\n"
        "1. Only suggest real Wikipedia articles that exist online.\n"
        "2. No duplicate titles.\n"
        "3. Base decision on both titles and summaries of existing articles.\n"
        "4. Maximum of 5 new titles if update is needed."
    ),
)

# Agent 3: Document Retrieval Agent
document_retrieve_agent = create_agent(
    model=llm,
    response_format=ToolStrategy(MultipleRetrievalDecision),
    system_prompt=(
        "You are a Document Retrieve Agent. Do NOT answer the user query yourself.\n"
        "Input: user query + current documents.\n"
        "Task: Decide if additional documents are needed.\n"
        "If current documents are insufficient:\n"
        "- set should_retrieve = true\n"
        "- provide reason for retrieval\n"
        "If documents are sufficient:\n"
        "- set should_retrieve = false\n"
        "- provide reason for no retrieval\n"
        "Rules:\n"
        "1. Only retrieve relevant documents.\n"
    ),
)

# Generator
generator = create_agent(
    llm,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
    system_prompt="You are a factual assistant that generates answers based on retrieved Wikipedia articles."
    "When answering, cite the Wikipedia article titles with URLs in parentheses."
    "Keep your answers concise and to the point.",
)

# --------------------------------------------------------
# Agent Run Function
# --------------------------------------------------------


def query_rewrite_run() -> str:
    global all_time_state

    print(all_time_state["chat_history"])

    if "updated_query" in all_time_state:
        all_time_state.pop("updated_query")
    if "new_titles" in all_time_state:
        all_time_state.pop("new_titles")

    all_time_state["final_k"] = 5
    all_time_state["should_retrieve"] = True

    result = query_rewrite_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "User Query: \n"
                    + all_time_state["query"]
                    + "\n\n Chat History: \n"
                    + str(all_time_state["chat_history"]),
                }
            ]
        }
    )

    reasoning = (
        "Rewrite is "
        + str(result["structured_response"].should_rewrite)
        + ". "
        + result["structured_response"].reason
    )
    all_time_state["updated_query"] = result["structured_response"].rewritten_query

    return reasoning


def knowledge_base_update_run() -> str:
    global all_time_state

    existing_articles = doc_store.get_all_articles()

    articles = [
        {
            "title": article["title"],
            "summary": article["summary"][:300],
        }
        for article in existing_articles
    ]

    result = knowledge_update_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "User Query: " + all_time_state["updated_query"]
                        if all_time_state.get("updated_query")
                        else all_time_state["query"]
                        + "\n\nExisting articles: \n"
                        + str(articles)
                    ),
                }
            ]
        }
    )

    if result["structured_response"].should_update:
        if len(result["structured_response"].new_titles) > 5:
            result["structured_response"].new_titles = result[
                "structured_response"
            ].new_titles[:5]
        wikipedia_tool(titles=result["structured_response"].new_titles or [])
        all_time_state["is_knowledge_updated"] = True
        all_time_state["new_titles"] = result["structured_response"].new_titles
    else:
        all_time_state["is_knowledge_updated"] = False

    reasoning = (
        "Knowledge base updates is "
        + str(result["structured_response"].should_update)
        + ". "
        + result["structured_response"].reason
    )

    return reasoning


def multiple_retrieval_run() -> str:
    global all_time_state

    current_k = 5
    reasoning = ""

    while all_time_state["should_retrieve"]:
        retrieved_docs = retrieve_tool(
            query=(
                all_time_state["updated_query"]
                if all_time_state.get("updated_query")
                else all_time_state["query"]
            ),
            k=current_k,
        )

        all_time_state["retrieved_documents"] = retrieved_docs

        result = document_retrieve_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "User Query: " + all_time_state["updated_query"]
                            if all_time_state.get("updated_query")
                            else all_time_state["query"]
                            + "\n\nCurrent Documents: \n"
                            + str(all_time_state["retrieved_documents"])
                        ),
                    }
                ]
            }
        )

        all_time_state["should_retrieve"] = result["structured_response"].should_retrieve

        if current_k == 5:
            reasoning = (
                "Multiple retrieval is "
                + str(result["structured_response"].should_retrieve)
                + ". "
                + result["structured_response"].reason
            )
            
        if result["structured_response"].should_retrieve is False or current_k == 30:
            all_time_state["final_k"] = current_k
            break

        current_k += 5

    return reasoning


# --------------------------------------------------------
# Generator Run Function
# --------------------------------------------------------


def generator_run():
    def generation():
        global all_time_state

        prompt = f"""
            User Query:
            {all_time_state['query']}

            Retrieved Documents:
            {all_time_state.get('retrieved_documents', [])}

            Answer:
        """

        response = generator.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            {"configurable": {"thread_id": "1"}},
        )

        final_text = response["messages"][-1].content

        chat_history = []
        for message in response["messages"]:
            role = message.type.upper()
            content = message.content
            if role == "HUMAN":
                if "User Query:" in content:
                    query_text = content.split("User Query:")[1].split(
                        "Retrieved Documents:"
                    )[0]
                    query_text = query_text.strip()
                    chat_history.append(("HUMAN", query_text))
            elif role == "AI":
                chat_history.append(("AI", message.content))

        all_time_state["chat_history"] = chat_history

        for line in final_text.splitlines():
            yield line + "\n"

    return generation()
