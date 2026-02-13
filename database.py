import os
from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_models.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SavedModel(Base):
    __tablename__ = "saved_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    epsilon = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    data = Column(LargeBinary)  # We will store the pickled agent here as binary blobs

# 4. Create Tables
def init_db():
    Base.metadata.create_all(bind=engine)