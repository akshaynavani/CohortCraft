"""Thin SQLite helpers: create schema, load DataFrames, run SQL files."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from cohortcraft.config import SQL_DIR


def connect(db_path: str | Path) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def read_sql_file(name: str, **params: str) -> str:
    """Read ``sql/<name>`` and substitute ``{param}`` placeholders.

    Only trusted, internal values (e.g. the snapshot date) are substituted.
    """
    text = (SQL_DIR / name).read_text()
    for key, value in params.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(read_sql_file("schema.sql"))


def load_tables(conn: sqlite3.Connection, tables: dict[str, pd.DataFrame]) -> None:
    """Create the schema and append each DataFrame into its table."""
    create_schema(conn)
    for name, df in tables.items():
        df.to_sql(name, conn, if_exists="append", index=False)
    conn.commit()


def query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def run_script_query(conn: sqlite3.Connection, name: str, **params: str) -> pd.DataFrame:
    """Execute a SQL file whose final statement is a SELECT and return it.

    Statements before the last one (e.g. temp views) are run as a script.
    """
    text = read_sql_file(name, **params)
    statements = [s.strip() for s in text.split(";") if s.strip()]
    if len(statements) > 1:
        conn.executescript(";\n".join(statements[:-1]) + ";")
    return pd.read_sql_query(statements[-1], conn)
