from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import RandomAgent

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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