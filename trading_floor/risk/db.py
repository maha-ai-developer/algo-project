# at bottom of risk/db.py
import sqlite3
from datetime import date

def get_daily_pnl(target_date: str = None):
    if target_date is None:
        target_date = date.today().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ts, symbol, side, qty, price, strategy, mode
        FROM trades
        WHERE substr(ts, 1, 10) = ?
        ORDER BY ts
        """,
        (target_date,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows
