from datetime import datetime
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def fetch_candidates(
    conn: sqlite3.Connection,
    jd_hash: Optional[str] = None,
    email: Optional[str] = None,
    limit: Optional[int] = None,
    order_by: str = "score DESC, created_at DESC",
) -> List[Tuple]:
    """Fetch candidates with optional filters.

    Returns rows as tuples matching:
    (id, name, email, phone, score, file_path, jd_hash, created_at)
    """
    cur = conn.cursor()
    query = "SELECT id, name, email, phone, score, file_path, jd_hash, created_at FROM candidates"
    conditions = []
    params: list = []
    if jd_hash:
        conditions.append("jd_hash = ?")
        params.append(jd_hash)
    if email:
        conditions.append("email LIKE ?")
        params.append(f"%{email}%")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    if order_by:
        query += f" ORDER BY {order_by}"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    cur.execute(query, tuple(params))
    return cur.fetchall()


def delete_candidate(conn: sqlite3.Connection, candidate_id: int) -> int:
    """Delete a candidate by id. Returns the number of rows deleted."""
    cur = conn.cursor()
    cur.execute("DELETE FROM candidates WHERE id=?", (candidate_id,))
    conn.commit()
    return cur.rowcount
