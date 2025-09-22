import sqlite3, re

# same CLEAN_PATTERNS and clean_text as in the ingester
CLEAN_PATTERNS = [
    r"\*P\d{5}A\d{3,4}\*", r"DO NOT WRITE IN THIS AREA", r"AERA SIHT NI ETIRW TON OD",
    r"Turn over", r"TOTAL FOR PAPER IS \d+ MARKS",
    r"Please check the examination details.*?You must have:.*?Instructions",
]
def clean_text(t: str) -> str:
    if not t: return ""
    for pat in CLEAN_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE | re.DOTALL)
    return " ".join(t.split())

TOPIC_RULES = {
    "Algebra": ["simplify","expand","factor","solve","equation","simultaneous","indices","powers","substitute","collect like terms"],
    "Inequalities": ["inequal","<",">"],
    "Graphs": ["graph","gradient","y=mx+c","line of best fit","coordinates","plot","table of values"],
    "Sequences": ["sequence","term","nth","arithmetic","geometric","fibonacci"],
    "Ratio & Proportion": ["ratio","proportion","direct proportion","inverse proportion","best buy","scale diagram"],
    "Geometry": ["angle","triangle","circle","perimeter","volume","polygon","parallel","perpendicular","pythag"],
    "Trigonometry": ["sin","cos","tan","trigon","hypotenuse","opposite","adjacent"],
    "Probability": ["probability","tree diagram","mutually exclusive","independent","sample space"],
    "Statistics": ["mean","median","mode","range","histogram","cumulative frequency","box plot","quartile","frequency table","pictogram"],
    "Transformations": ["enlarge","scale factor","reflect","rotate","translate","transformation","enlargement"],
    "Vectors": ["vector","column vector","magnitude"],
    "Equations": ["quadratic","roots","equation of the line","simultaneous equations"],
    "Functions": ["function","f(x)","composite","inverse function"],
    "Number": ["percentage","fraction","decimal","prime","hcf","lcm","standard form","rounding","estimate","density","unit rate"]
}
def guess_topic(t: str):
    s = (t or "").lower()
    best, hits = None, 0
    for topic, kws in TOPIC_RULES.items():
        c = sum(1 for k in kws if k in s)
        if c > hits:
            hits, best = c, topic
    return best if hits > 0 else None

DB_PATH = "app.db"
conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("SELECT id, text_snippet FROM questions")
rows = cur.fetchall()
updated = 0
for r in rows:
    cleaned = clean_text(r["text_snippet"] or "")
    topic = guess_topic(cleaned)
    cur.execute("UPDATE questions SET text_snippet=?, topic=? WHERE id=?", (cleaned, topic, r["id"]))
    updated += 1
conn.commit(); conn.close()
print(f"Retagged {updated} questions.")