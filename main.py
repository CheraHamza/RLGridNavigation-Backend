from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import RandomAgent, QLearningAgent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# agent = RandomAgent()

# class State(BaseModel):
#     position: list[int]
#     target: list[int]

# class ActionResponse(BaseModel):
#     action: str


# @app.post("/act", response_model=ActionResponse)
# def act(state: State):
#     action = agent.act(state.model_dump())
#     return {"action": action}


agent = QLearningAgent()

class StepData(BaseModel):
    position: list[int]
    target: list[int]    # Included but not used by Q-Learning yet
    reward: float
    done: bool

class ActionResponse(BaseModel):
    action: str
    epsilon: float


@app.post("/act", response_model=ActionResponse)
def act(data: StepData):
    # 1. Learn from the Previous step (using current reward)
    agent.learn(data.position, data.reward, data.done)

    # 2. If episode is done, we don't need a new action
    if data.done:
        return {"action": "stop", "epsilon": agent.epsilon}
    
    # 3. Choose Next action based on current state
    action = agent.choose_action(data.position)

    # 4. Remember this state/action for the next learning step
    agent.update_memory(data.position, action)

    return {"action": action, "epsilon": agent.epsilon}