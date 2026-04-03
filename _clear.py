import sqlite3, os
db = sqlite3.connect(os.path.expanduser("~/.applypilot/applypilot.db"))
c = db.execute("UPDATE jobs SET apply_status = NULL, agent_id = NULL WHERE apply_status = 'in_progress'")
db.commit()
print(f"Cleared {c.rowcount} stale locks")
