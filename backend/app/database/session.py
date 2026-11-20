import sqlite3
from . import __name__
from ..config import DATABASE_PATH

def connect():
    db=sqlite3.connect(DATABASE_PATH); db.row_factory=sqlite3.Row
    db.execute("pragma foreign_keys=on")
    return db

def init_db():
    DATABASE_PATH.parent.mkdir(parents=True,exist_ok=True)
    with connect() as db:
        db.execute("create table if not exists analyses(id text primary key, filename text not null, path text not null, sample_rate integer not null, duration real not null, created_at text not null, status text not null, result text)")
