from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DATABASE_ECHO, DEFAULT_DATABASE_URL

Base = declarative_base()


def get_engine(database_url: str | None = None):
    url = database_url or DEFAULT_DATABASE_URL
    return create_engine(url, echo=DATABASE_ECHO, future=True)


def init_db(database_url: str | None = None):
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    return engine


SessionLocal = sessionmaker(autocommit=False, autoflush=False, future=True)

__all__ = ["Base", "SessionLocal", "get_engine", "init_db"]
