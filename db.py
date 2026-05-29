"""
Carrot Bot — SQLite persistence for dry-run portfolio.
Stores positions, trades, and cash across bot restarts.
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "carrot.db"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


class Persistence:
    """Thread-safe SQLite persistence for portfolio state."""

    def __init__(self, db_path: str = None):
        self._db_path = db_path or str(DB_PATH)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path, timeout=10)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init_db(self):
        c = self._conn()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS kv (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS positions (
                pos_id      TEXT PRIMARY KEY,
                market_id   TEXT NOT NULL,
                shares      REAL NOT NULL,
                price       REAL NOT NULL,
                cost        REAL NOT NULL,
                outcome     TEXT DEFAULT 'YES',
                market_title TEXT DEFAULT '',
                event_slug  TEXT DEFAULT '',
                end_date    TEXT DEFAULT '',
                opened_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id   TEXT NOT NULL,
                market_title TEXT DEFAULT '',
                event_slug  TEXT DEFAULT '',
                entry_price REAL,
                exit_price  REAL,
                shares      REAL,
                cost        REAL,
                pnl         REAL,
                pnl_pct     REAL,
                closed      INTEGER DEFAULT 1,
                closed_at   TEXT
            );
            CREATE TABLE IF NOT EXISTS history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                cash        REAL,
                invested    REAL,
                value       REAL,
                pnl         REAL,
                positions   INTEGER,
                trades      INTEGER,
                accuracy    REAL,
                model_version TEXT,
                training    INTEGER
            );
        """)
        c.commit()

    # ── Key-value (cash, total_trades, etc.) ───────────────────────────────

    def get_kv(self, key: str, default=None):
        row = self._conn().execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_kv(self, key: str, value):
        self._conn().execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
            (key, json.dumps(value, default=str)),
        )
        self._conn().commit()

    # ── Portfolio snapshot ─────────────────────────────────────────────────

    def load_portfolio(self) -> dict:
        """Load full portfolio state. Returns dict with cash, positions, trades."""
        cash = self.get_kv("cash")
        initial_cash = self.get_kv("initial_cash")
        total_trades = self.get_kv("total_trades", 0)

        rows = self._conn().execute("SELECT * FROM positions").fetchall()
        positions = {}
        for r in rows:
            pos = dict(r)
            pid = pos.pop("pos_id")
            pos["opened_at"] = pos["opened_at"]
            positions[pid] = pos

        trades = [dict(r) for r in self._conn().execute("SELECT * FROM trades ORDER BY id").fetchall()]

        return {
            "cash": cash,
            "initial_cash": initial_cash,
            "total_trades": total_trades,
            "positions": positions,
            "trades": trades,
        }

    def save_portfolio(self, portfolio):
        """Save full portfolio state atomically. Appends new trades only (never deletes history)."""
        c = self._conn()

        # Read how many trades we've already saved before modifying anything
        already_saved = self.get_kv("trades_saved_count", 0)

        try:
            c.execute("BEGIN IMMEDIATE")
            c.execute("DELETE FROM positions")
            c.execute("DELETE FROM kv WHERE key IN ('cash','initial_cash','total_trades','trades_saved_count')")

            c.execute("INSERT INTO kv (key,value) VALUES ('cash',?)", (json.dumps(portfolio.cash),))
            c.execute("INSERT INTO kv (key,value) VALUES ('initial_cash',?)", (json.dumps(portfolio.initial_cash),))
            c.execute("INSERT INTO kv (key,value) VALUES ('total_trades',?)", (json.dumps(portfolio.total_trades),))

            for pid, pos in portfolio.positions.items():
                opened = pos.get("opened_at", "")
                if hasattr(opened, "isoformat"):
                    opened = opened.isoformat()
                c.execute(
                    "INSERT INTO positions (pos_id,market_id,shares,price,cost,outcome,market_title,event_slug,end_date,opened_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (pid, pos["market_id"], pos["shares"], pos["price"], pos["cost"],
                     pos.get("outcome", "YES"), pos.get("market_title", ""),
                     pos.get("event_slug", ""), pos.get("end_date", ""), opened),
                )

            # Append-only: only insert trades beyond what was already saved
            trade_list = portfolio.trade_history
            new_trades = trade_list[already_saved:]
            for t in new_trades:
                closed_at = t.get("closed_at", "")
                if hasattr(closed_at, "isoformat"):
                    closed_at = closed_at.isoformat()
                c.execute(
                    "INSERT INTO trades (market_id,market_title,event_slug,entry_price,exit_price,shares,cost,pnl,pnl_pct,closed,closed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (t.get("market_id", ""), t.get("market_title", ""), t.get("event_slug", ""),
                     t.get("entry_price", 0), t.get("exit_price", 0), t.get("shares", 0),
                     t.get("cost", 0), t.get("pnl", 0), t.get("pnl_pct", 0),
                     1 if t.get("closed") else 0, closed_at),
                )

            c.execute("INSERT INTO kv (key,value) VALUES ('trades_saved_count',?)",
                      (json.dumps(len(trade_list)),))
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    # ── History snapshots ──────────────────────────────────────────────────

    def save_history_snapshot(self, entry: dict):
        c = self._conn()
        c.execute(
            "INSERT INTO history (ts,cash,invested,value,pnl,positions,trades,accuracy,model_version,training) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (entry["ts"], entry["cash"], entry["invested"], entry["value"],
             entry["pnl"], entry["positions"], entry["trades"],
             entry.get("accuracy"), entry.get("model_version", ""),
             1 if entry.get("training") else 0),
        )
        c.commit()

    def load_history(self, limit: int = 500) -> list:
        rows = self._conn().execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def trim_history(self, keep: int = 500):
        self._conn().execute(
            "DELETE FROM history WHERE id NOT IN (SELECT id FROM history ORDER BY id DESC LIMIT ?)",
            (keep,),
        )
        self._conn().commit()
