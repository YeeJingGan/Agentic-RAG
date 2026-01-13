from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi import FastAPI
import agent 

app = FastAPI()


class PromptRequest(BaseModel):
    query: str


@app.post("/agent1")
async def execute_agent1(request: PromptRequest):
    agent.all_time_state["query"] = request.query

    reasoning = agent.query_rewrite_run()
    return reasoning


@app.get("/agent2")
async def execute_agent2():
    reasoning = agent.knowledge_base_update_run()
    return reasoning


@app.get("/agent3")
async def execute_agent3():
    reasoning = agent.multiple_retrieval_run()
    return reasoning

@app.get("/state")
async def get_state():
    return agent.all_time_state

@app.get("/query")
async def query_endpoint():
    return StreamingResponse(agent.generator_run(), media_type="text/plain")
