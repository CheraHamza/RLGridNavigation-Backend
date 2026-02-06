from fastapi import FastAPI
from pydantic import BaseModel
from agent import RandomAgent

app = FastAPI()
agent = RandomAgent()

class State(BaseModel):
    position: list[int]
    target: list[int]

class ActionResponse(BaseModel):
    action: str


@app.post("/act", response_model=ActionResponse)
def act(state: State):
    action = agent.act(state.model_dump())
    return {"action": action}