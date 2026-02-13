import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from agent import QLearningAgent
from database import SessionLocal, init_db, SavedModel


init_db()

app = FastAPI()

cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:5173")
cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
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
    reward: float
    done: bool

class ActionResponse(BaseModel):
    action: str
    epsilon: float

class ModelCreate(BaseModel):
    name: str

class ModelList(BaseModel):
    id: int
    name: str
    epsilon: float
    created_at: datetime


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


@app.get("/models", response_model=List[ModelList])
def list_models(db: Session = Depends(get_db)):
    """Return a list of all saved models."""
    return db.query(SavedModel).all()

@app.post("/models")
def save_model(model_input: ModelCreate, db: Session = Depends(get_db)):
    """Save the current agent to the DB."""
    binary_data = agent.to_bytes()
    
    new_model = SavedModel(
        name=model_input.name,
        epsilon=agent.epsilon,
        data=binary_data
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
    return {"status": "loaded", "epsilon": agent.epsilon, "name": saved_model.name}

@app.delete("/models/{model_id}")
def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Delete a model."""
    saved_model = db.query(SavedModel).filter(SavedModel.id == model_id).first()
    if not saved_model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    db.delete(saved_model)
    db.commit()
    return {"status": "deleted"}