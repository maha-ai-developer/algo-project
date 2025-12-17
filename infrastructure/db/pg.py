# db/pg.py
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()
_engine = None
SessionLocal = None

def init_db(db_url: str):
    global _engine, SessionLocal
    _engine = create_engine(db_url, echo=False, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(_engine)

def get_session():
    if SessionLocal is None:
        raise RuntimeError("DB not initialized. Call init_db(db_url) first.")
    return SessionLocal()

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(32), index=True)
    side = Column(String(4))        # BUY/SELL
    qty = Column(Integer)
    price = Column(Float)
    product = Column(String(8))     # MIS/CNC/etc.
    pnl = Column(Float, nullable=True)
    sim = Column(Boolean, default=True)
    strategy = Column(String(64), default="combined_stack")

class Bar(Base):
    __tablename__ = "bars"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True)
    symbol = Column(String(32), index=True)
    timeframe = Column(String(8))
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

class PositionSnapshot(Base):
    __tablename__ = "position_snapshots"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(32), index=True)
    qty = Column(Integer)
    avg_price = Column(Float)
    product = Column(String(8))
    net_pnl = Column(Float)
