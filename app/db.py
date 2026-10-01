"""
Database helpers and the seed data.

The seed is the "clean world" every case starts from. It is built once,
then copied per case (see bench/env.py), so cases never share state.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from pathlib import Path

SCHEMA = Path(__file__).with_name("schema.sql")
_ITERATIONS = 100_000

USERS = [
    # email, password, display name
    ("alice@example.com", "password123", "Alice Chen"),
    ("bob@example.com", "hunter22", "Bob Rivera"),
]

PRODUCTS = [
    # sku, name, price in cents, stock
    ("SHIRT-BLU", "Blue Oxford Shirt", 4800, 20),
    ("SHIRT-WHT", "White Oxford Shirt", 4800, 20),
    ("SOCK-WOOL", "Wool Socks", 1400, 50),
    ("TOTE-CNV", "Canvas Tote", 2200, 15),
    ("MUG-ENML", "Enamel Mug", 1800, 30),
    ("CAP-BLK", "Black Cap", 2500, 10),
    ("BELT-LTH", "Leather Belt", 5500, 8),
    ("SCARF-GRY", "Grey Scarf", 3900, 12),
]


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split("$", 1)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), _ITERATIONS
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


def connect(path: str | Path) -> sqlite3.Connection:
    # check_same_thread=False: FastAPI may open the connection in one worker
    # thread and use it in another. Safe because each request gets its own.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: str | Path) -> Path:
    """Create a fresh database at `path` with schema + seed data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    conn = connect(path)
    try:
        conn.executescript(SCHEMA.read_text())
        conn.executemany(
            "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
            [(e, hash_password(p), n) for e, p, n in USERS],
        )
        conn.executemany(
            "INSERT INTO products (sku, name, price_cents, stock) VALUES (?, ?, ?, ?)",
            PRODUCTS,
        )
        conn.commit()
    finally:
        conn.close()
    return path
