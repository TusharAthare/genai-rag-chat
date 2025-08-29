from datetime import datetime
import sqlite3
from pathlib import Path
from typing import Dict

DB_PATH = Path("recruiting.db")


def init_db():
    """Initialize the SQLite database and ensure required schema exists."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            score REAL,
            file_path TEXT,
            jd_hash TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_jd ON candidates(jd_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_emailjd ON candidates(email, jd_hash)")
    conn.commit()
    return conn


def candidate_exists(conn: sqlite3.Connection, email: str, jd_hash: str) -> bool:
    """Check if a candidate (by email) is already saved for a JD."""
    if not email:
        return False
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM candidates WHERE email=? AND jd_hash=? LIMIT 1", (email, jd_hash))
    return cur.fetchone() is not None


def save_candidate(conn: sqlite3.Connection, info: Dict[str, str], score: float, file_path: Path, jd_hash: str):
    """Insert a candidate record into the database."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO candidates(name, email, phone, score, file_path, jd_hash, created_at) VALUES (?,?,?,?,?,?,?)",
        (
            info.get("name", ""),
            info.get("email", ""),
            info.get("phone", ""),
            float(score),
            str(file_path),
            jd_hash,
            datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()

