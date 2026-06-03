"""Hardened SQLite event log (Phase 5).

Every write goes through ONE dedicated writer thread fed by a queue, and the DB
runs in WAL mode — so the real-time loop never blocks on disk and the dashboard
(a separate reader) never collides with the writer ("database is locked" gone).
Public API (log/close) is unchanged, so app.py doesn't change.
"""
import sqlite3
import threading
import queue
from datetime import datetime


class Storage:
    _SENTINEL = object()

    def __init__(self, db_path):
        self.path = db_path
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._writer_loop, name="db-writer", daemon=True)
        self._thread.start()

    def _connect(self):
        # Only the writer thread ever touches this connection.
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")     # readers don't block the writer
        conn.execute("PRAGMA synchronous=NORMAL;")   # WAL-safe and much faster
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS traffic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                camera TEXT,
                event_type TEXT,
                occupancy INTEGER,
                clip_path TEXT,
                snapshot_path TEXT
            )
            """
        )
        for col in ("camera TEXT", "clip_path TEXT", "snapshot_path TEXT"):
            try:
                conn.execute(f"ALTER TABLE traffic_events ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.commit()
        return conn

    def _writer_loop(self):
        conn = self._connect()
        try:
            while True:
                item = self._q.get()
                if item is self._SENTINEL:
                    break
                ts, camera, event_type, occupancy, clip_path, snapshot_path = item
                try:
                    conn.execute(
                        "INSERT INTO traffic_events "
                        "(timestamp, camera, event_type, occupancy, clip_path, snapshot_path) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (ts, camera, event_type, occupancy, clip_path, snapshot_path),
                    )
                    conn.commit()              # keep write transactions tiny
                except sqlite3.OperationalError as e:
                    print(f"[Storage] write failed: {e}")   # never let the writer die
        finally:
            conn.close()

    def log(self, camera, event_type, occupancy, clip_path=None, snapshot_path=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._q.put((ts, camera, event_type, occupancy, clip_path, snapshot_path))
        return ts

    def close(self):
        self._q.put(self._SENTINEL)
        self._thread.join(timeout=5.0)
