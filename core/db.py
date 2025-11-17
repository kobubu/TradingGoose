# core/db.py
import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extensions import connection as _PGConnection

import logging
logger = logging.getLogger(__name__)


def _build_dsn() -> str:
    dsn_env = os.getenv("PG_DSN")
    if dsn_env:
        return dsn_env

    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db   = os.getenv("PG_DB", "yourtrader")
    user = os.getenv("PG_USER", "yourtrader")
    pwd  = os.getenv("PG_PASSWORD", "")

    # классический DSN для psycopg2
    return f"dbname={db} user={user} password={pwd} host={host} port={port}"


PG_DSN = _build_dsn()


@contextmanager
def db() -> _PGConnection:
    conn = None
    try:
        logger.info("Connecting to Postgres with DSN: %s", PG_DSN)
        conn = psycopg2.connect(PG_DSN)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.exception("Postgres connection error: %s", e)
        raise
    finally:
        if conn:
            conn.close()