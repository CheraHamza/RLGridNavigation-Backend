import os
import logging
from typing import Optional
from sqlalchemy import create_engine, String
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from datetime import datetime

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_models.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Build engine kwargs depending on the DB backend
_is_sqlite = "sqlite" in DATABASE_URL

if _is_sqlite:
    _connect_args: dict = {"check_same_thread": False}
    _engine_kwargs: dict = {}
else:
    # PostgreSQL (e.g. Aiven) — enforce a connection timeout so the app
    # doesn't hang forever during cold-starts, and require SSL.
    _connect_args = {
        "connect_timeout": 10,
        "sslmode": "require",
    }
    _engine_kwargs = {
        "pool_pre_ping": True,      # recycle stale connections automatically
        "pool_recycle": 300,        # recycle connections every 5 min
    }

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    **_engine_kwargs,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class SavedModel(Base):
    __tablename__ = "saved_models"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    epsilon: Mapped[float] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    data: Mapped[bytes] = mapped_column()  # pickled agent stored as binary blob
    environment_config: Mapped[Optional[str]] = mapped_column(String, default="{}")  # JSON string of env config


def init_db():
    """Create tables.  Non-fatal on failure so the HTTP server can still start."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created / verified.")
    except Exception as exc:
        logger.warning("init_db failed (will retry on first request): %s", exc)