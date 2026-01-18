import os
from pathlib import Path
from dotenv import load_dotenv
from rag.chunker import Chunker
from langchain.tools import tool
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
from pydantic import BaseModel, Field
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
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]}


@tool(description="Update the all_time_state with the given key and value.")
def update_state(key: str, value: str):
    global all_time_state

    all_time_state[key] = value


@tool(description="Retrieve Wikipedia articles by their titles when the existing articles are not sufficient to answer the query. Titles should be distinct.")
def wikipedia_tool(titles: List[str]):
    existing_articles = doc_store.get_all_articles()
    existing_titles = {article["title"] for article in existing_articles}
    for title in titles:
        if title in existing_titles:
            titles.remove(title)
    wiki_processor.process(titles=titles)


@tool(description="Retrieve documents for a given query when the current documents do not contain any information about the query.")
def retrieve_tool(query: str, k: int) -> List[str]:
    _, _, _, parent_docs, articles = retriever.retrieve(
        query=query, k=k
    )
    
    all_time_state['retrieved_documents'] = parent_docs

    return parent_docs


# --------------------------------------------------------
# Define Structured Output Models
# --------------------------------------------------------
class Reason(BaseModel):
    reason: str | None = Field(
        ...,
        description="The reason for executing the tool if tool is used else mention why not.",
    )


# --------------------------------------------------------
# Model and Agent Creation
# ---------------------------------------------------------

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# Agent 1: Query Rewrite Agent
query_rewrite_agent = create_agent(
    model=llm,
    tools=[update_state],
    response_format=ToolStrategy(Reason),
    system_prompt="You are a Wikipedia Query Optimization Agent. Do not answer to the query, follow the instructions carefully. "
    "1. Comprehend the user query and the chat history. "
    "2. Analyse whether the user query is optimized to retrieve documents from Wikpedia articles."
    "3. If user query is not optimized, you must use the update_state tool to update the all_time_state with the rewritten query "
    'where key = "updated_query".',
)

# Agent 2: Knowledge Base Update Agent
knowledge_update_agent = create_agent(
    model=llm,
    tools=[wikipedia_tool, update_state],
    response_format=ToolStrategy(Reason),
    system_prompt='You are a Knowledge Base Update Agent. Do not answer to the query, follow the instructions carefully. ' \
    '1. Analyse whether the existing articles provided are sufficient to answer the query. ' \
    '2. When existing articles do not answer the query, you must use the wikipedia_tool to retrieve additional Wikipedia articles that are not in the existing articles.' \
    '3. Use the update_state tool to update the all_time_state with key = "is_knowledge_updated" and value = "true" when wikipedia_tool is used else "false". ')

# Agent 3: Document Retrieval Agent
document_retrieve_agent = create_agent(
    model=llm,
    tools=[retrieve_tool, update_state],
    response_format=ToolStrategy(Reason),
    system_prompt='You are a Document Retrieve Agent. Do not answer to the query, follow the instructions carefully. ' \
    '1. Assess whether the current documents answer the query.' \
    '2. If insufficient, loop at most 3 times for the following procedures: ' \
    '   a. Use the `retrieve_tool` to get additional relevant documents.' \
    '   b. Each time, use an integer value higher than 10 for \'k\' in the `retrieve_tool`. Over iterations, gradually increase \'k\' by at least 5 each time.'
    '   c. Evaluate the newly retrieved documents.'
    '   d. Stop if the documents sufficiently answer the query.' \
    '3. Use the update_state tool to update the all_time_state with key = "k" and value = k used in retrieve_tool. ')

# Generator 
generator = create_agent(
    llm,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
    system_prompt="You are a factual assistant that generates answers based on retrieved Wikipedia articles." \
    "When answering, cite the Wikipedia article titles with URLs in parentheses." \
    "Keep your answers concise and to the point.",
)

# --------------------------------------------------------
# Agent Run Function
# --------------------------------------------------------

def query_rewrite_run() -> str:
    global all_time_state

    if 'updated_query' in all_time_state:
        all_time_state.pop('updated_query')


    result = query_rewrite_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "User Query: \n"
                    + all_time_state["query"]
                    + "\n\n Chat History: \n"
                    + str(all_time_state["chat_history"])
                }
            ]
        }
    )

    reasoning = result['structured_response'].reason

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
    
    result = knowledge_update_agent.invoke({
    "messages": [{"role": "user", "content": 'User Query: ' + all_time_state["query"] + 
                  '\n\nExisting articles: \n' + str(articles)}]})
    
    reasoning = result['structured_response'].reason

    return reasoning


def multiple_retrieval_run() -> str:
    global all_time_state

    # For first retrieval
    _, _, _, documents, _ = retriever.retrieve(query=all_time_state['updated_query'] if all_time_state.get('updated_query') else all_time_state['query'])
    all_time_state['retrieved_documents'] = documents

    result = document_retrieve_agent.invoke({
    "messages": [{"role": "user", "content": 'User Query: ' + all_time_state["updated_query"] if all_time_state.get("updated_query") else all_time_state["query"] + 
                  '\n\nCurrent Documents: \n' + str(all_time_state['retrieved_documents'])}]})
    
    reasoning = result['structured_response'].reason

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

        history = []
        for message in response["messages"]:
            role = message.type.upper() 
            history.append((role, message.content))
        
        all_time_state['chat_history'] = history

        for line in final_text.splitlines():
            yield line + "\n"

    return generation()
