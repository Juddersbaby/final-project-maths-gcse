import sqlite3
conn = sqlite3.connect("app.db")
cur = conn.cursor()
cur.execute("UPDATE questions SET page_start = page_end WHERE qno = 1 AND page_start < page_end")
print("Rows updated:", cur.rowcount)
conn.commit()
conn.close()
