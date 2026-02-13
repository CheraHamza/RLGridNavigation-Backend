from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import json
from agent import QLearningAgent
from database import SessionLocal, init_db, SavedModel
from environment import GridWorld


init_db()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


agent = QLearningAgent()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class StepData(BaseModel):
    position: list[int]
    target: list[int]    # Included but not used by Q-Learning yet
    obstacles: list[list[int]] = []
    reward: float
    done: bool

class ActionResponse(BaseModel):
    action: str
    epsilon: float

class EnvironmentConfig(BaseModel):
    height: int = 10
    width: int = 10
    starting_position: list[int] = [0, 0]
    target_position: list[int] = [8, 8]
    obstacles: list[list[int]] = []

class ModelCreate(BaseModel):
    name: str
    environment: EnvironmentConfig = EnvironmentConfig()

class ModelList(BaseModel):
    id: int
    name: str
    epsilon: float
    created_at: datetime
    environment: Optional[EnvironmentConfig] = None

class TrainRequest(BaseModel):
    episodes: int = 500
    height: int = 10
    width: int = 10
    starting_position: list[int] = [0, 0]
    target_position: list[int] = [8, 8]
    obstacles: list[list[int]] = []

class EpisodeResult(BaseModel):
    episode: int
    steps: int
    total_reward: float
    reached_target: bool

class TrainResponse(BaseModel):
    episodes_trained: int
    epsilon: float
    results: list[EpisodeResult]


@app.get("/health")
def health():
    """Lightweight liveness probe – used by the frontend to detect cold-start."""
    return {"status": "ok"}


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


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    """
    Run many episodes entirely server-side in a tight loop.
    No HTTP round-trips or UI updates per step – orders of magnitude faster.
    """
    env = GridWorld(
        height=req.height,
        width=req.width,
        starting_position=req.starting_position,
        target_position=req.target_position,
        obstacles=req.obstacles,
    )

    results: list[EpisodeResult] = []

    for ep in range(1, req.episodes + 1):
        obs = env.reset()
        total_reward = 0.0
        reached_target = False

        # First step: no learning yet, just pick an action
        agent.learn(obs["position"], 0.0, False)  # no-op on first call (prev_state is None)
        action = agent.choose_action(obs["position"])
        agent.update_memory(obs["position"], action)

        while True:
            result = env.step(action)
            state = result["state"]
            reward = result["reward"]
            done = result["done"]
            total_reward += reward

            # Learn from this transition
            agent.learn(state["position"], reward, done)

            if done:
                reached_target = (
                    state["position"][0] == req.target_position[0]
                    and state["position"][1] == req.target_position[1]
                )
                break

            # Pick next action
            action = agent.choose_action(state["position"])
            agent.update_memory(state["position"], action)

        results.append(
            EpisodeResult(
                episode=ep,
                steps=result["steps"],
                total_reward=round(total_reward, 4),
                reached_target=reached_target,
            )
        )

    return TrainResponse(
        episodes_trained=req.episodes,
        epsilon=agent.epsilon,
        results=results,
    )


@app.get("/models", response_model=List[ModelList])
def list_models(db: Session = Depends(get_db)):
    """Return a list of all saved models."""
    rows = db.query(SavedModel).all()
    result = []
    for row in rows:
        env_config = None
        if row.environment_config:
            try:
                env_config = json.loads(row.environment_config)
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(ModelList(
            id=row.id,
            name=row.name,
            epsilon=row.epsilon,
            created_at=row.created_at,
            environment=env_config,
        ))
    return result

@app.post("/reset")
def reset_agent():
    """Reset the agent's Q-table and epsilon (e.g. when environment changes)."""
    agent.q_table = {}
    agent.epsilon = 1.0
    agent.prev_state = None
    agent.prev_action = None
    return {"status": "reset", "epsilon": agent.epsilon}


@app.post("/models")
def save_model(model_input: ModelCreate, db: Session = Depends(get_db)):
    """Save the current agent + its environment config to the DB."""
    binary_data = agent.to_bytes()
    env_json = model_input.environment.model_dump_json()
    
    new_model = SavedModel(
        name=model_input.name,
        epsilon=agent.epsilon,
        data=binary_data,
        environment_config=env_json,
    )
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return {"status": "saved", "id": new_model.id, "name": new_model.name}

@app.post("/models/{model_id}/load")
def load_model(model_id: int, db: Session = Depends(get_db)):
    """Load a specific model from DB into the active agent."""
    saved_model = db.query(SavedModel).filter(SavedModel.id == model_id).first()
    if not saved_model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    agent.from_bytes(saved_model.data)

    # Parse stored environment config
    env_config = None
    if saved_model.environment_config:
        try:
            env_config = json.loads(saved_model.environment_config)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "status": "loaded",
        "epsilon": agent.epsilon,
        "name": saved_model.name,
        "environment": env_config,
    }

@app.delete("/models/{model_id}")
def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Delete a model."""
    saved_model = db.query(SavedModel).filter(SavedModel.id == model_id).first()
    if not saved_model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    db.delete(saved_model)
    db.commit()
    return {"status": "deleted"}