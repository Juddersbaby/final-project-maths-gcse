import re, os, sys, argparse, sqlite3, requests, io, pdfplumber

DB_PATH = os.environ.get("DB_PATH", "app.db")

CLEAN_PATTERNS = [
    r"\*P\d{5}A\d{3,4}\*",                # e.g. *P69525A0224*
    r"DO NOT WRITE IN THIS AREA",         # repeated frames
    r"AERA SIHT NI ETIRW TON OD",         # mirrored text
    r"Turn over",                         # pagination
    r"TOTAL FOR PAPER IS \d+ MARKS",
    r"Please check the examination details.*?You must have:.*?Instructions",  # cover blurb
]

def clean_text(t: str) -> str:
    if not t: return ""
    for pat in CLEAN_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE | re.DOTALL)
    # collapse whitespace
    t = " ".join(t.split())
    return t

TOPIC_RULES = {
    "Algebra": [
        "simplify", "expand", "factor", "solve", "equation", "simultaneous",
        "indices", "powers", "substitute", "collect like terms"
    ],
    "Inequalities": ["inequal", "<", ">"],
    "Graphs": ["graph", "gradient", "y=mx+c", "line of best fit", "coordinates", "plot", "table of values"],
    "Sequences": ["sequence", "term", "nth", "arithmetic", "geometric", "fibonacci"],
    "Ratio & Proportion": ["ratio", "proportion", "direct proportion", "inverse proportion", "best buy", "scale diagram"],
    "Geometry": ["angle", "triangle", "circle", "perimeter", "volume", "polygon", "parallel", "perpendicular", "pythag"],
    "Trigonometry": ["sin", "cos", "tan", "trigon", "hypotenuse", "opposite", "adjacent"],
    "Probability": ["probability", "tree diagram", "mutually exclusive", "independent", "sample space"],
    "Statistics": ["mean", "median", "mode", "range", "histogram", "cumulative frequency", "box plot", "quartile", "frequency table", "pictogram"],
    "Transformations": ["enlarge", "scale factor", "reflect", "rotate", "translate", "transformation", "enlargement"],
    "Vectors": ["vector", "column vector", "magnitude"],
    "Equations": ["quadratic", "roots", "equation of the line", "simultaneous equations"],
    "Functions": ["function", "f(x)", "composite", "inverse function"],
    "Number": ["percentage", "fraction", "decimal", "prime", "hcf", "lcm", "standard form", "rounding", "estimate", "density", "unit rate"]
}


def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            board TEXT, tier TEXT, series TEXT, paper_no INTEGER, calculator INTEGER, year INTEGER,
            pdf_url TEXT NOT NULL, markscheme_url TEXT,
            UNIQUE(board,tier,series,paper_no)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL, qno INTEGER NOT NULL, marks INTEGER,
            page_start INTEGER NOT NULL, page_end INTEGER NOT NULL,
            topic TEXT, difficulty INTEGER,
            text_snippet TEXT,
            FOREIGN KEY(paper_id) REFERENCES papers(id),
            UNIQUE(paper_id, qno)
        )
    """)
    # make sure text_snippet exists even if table already existed
    try:
        cur.execute("ALTER TABLE questions ADD COLUMN text_snippet TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

def parse_series_from_url(url: str):
    # mathsgenie pattern e.g. /1fnov2023.pdf, /2hjune2019.pdf
    m = re.search(r"/(\d)([fh])([a-z]+)?(\d{4})\.pdf$", url, flags=re.I)
    if not m:
        return None
    paper_no = int(m.group(1))
    tier = "Foundation" if m.group(2).lower() == "f" else "Higher"
    mon = (m.group(3) or "").lower()
    year = int(m.group(4))
    month_map = {
        "nov": "Nov", "november": "Nov",
        "june": "June", "may": "May",
        "jan": "Jan", "january": "Jan",
        "march": "March", "mar": "March"
    }
    series = f"{month_map.get(mon, mon.title())} {year}".strip()
    calculator = 0 if paper_no == 1 else 1
    return paper_no, tier, series, calculator, year

def extract_questions(pdf_bytes: bytes):
    """Return list of (qno, marks, page_start, page_end).
       First question start is clamped to its end page; others follow sequentially."""
    hits = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for qno, marks in re.findall(r"Total\s+for\s+Question\s+(\d+)\s+is\s+(\d+)\s+mark", text, flags=re.I):
                hits.append((int(qno), int(marks), i))  # (qno, marks, end_page)

    if not hits:
        return []

    hits.sort(key=lambda x: x[0])
    ranges = []
    prev_end = None
    for (qno, marks, end_page) in hits:
        if prev_end is None:
            # First question: start on the page where its "Total for Question ..." appears
            start = end_page
        else:
            # Subsequent questions start on the page after the previous question's end
            start = prev_end + 1
            if start > end_page:
                # Safety: never start after the end page
                start = end_page
        ranges.append((qno, marks, start, end_page))
        prev_end = end_page

    return ranges


def compact(text: str, limit=600):
    t = " ".join((text or "").split())
    return t[:limit]

def guess_topic(text: str):
    t = (text or "").lower()
    best, hits = None, 0
    for topic, kws in TOPIC_RULES.items():
        c = sum(1 for k in kws if k in t)
        if c > hits:
            hits, best = c, topic
    return best if hits > 0 else None

def add_paper(conn, board, tier, series, paper_no, calculator, year, pdf_url, markscheme_url=None):
    cur = conn.cursor()
    cur.execute("""INSERT OR IGNORE INTO papers(board,tier,series,paper_no,calculator,year,pdf_url,markscheme_url)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (board, tier, series, paper_no, int(calculator), year, pdf_url, markscheme_url))
    conn.commit()
    cur.execute("""SELECT id FROM papers WHERE board=? AND tier=? AND series=? AND paper_no=?""",
                (board, tier, series, paper_no))
    return cur.fetchone()[0]

def main():
    ap = argparse.ArgumentParser(description="Ingest a MathsGenie past paper (metadata only).")
    ap.add_argument("--url", required=True, help="Paper PDF URL, e.g. https://www.mathsgenie.co.uk/papers/1fnov2023.pdf")
    ap.add_argument("--board", default="Edexcel")
    ap.add_argument("--markscheme_url", default=None)
    args = ap.parse_args()

    meta = parse_series_from_url(args.url)
    if not meta:
        print("Could not parse tier/series/paper from URL; expected /<num><f|h><month><year>.pdf")
        sys.exit(1)
    paper_no, tier, series, calculator, year = meta

    print(f"Ingesting: {args.url}  [{series}, {tier}, Paper {paper_no}, calc={calculator}]")
    r = requests.get(args.url, timeout=60); r.raise_for_status()
    pdf_bytes = r.content

    ranges = extract_questions(pdf_bytes)
    if not ranges:
        print("No 'Total for Question … is … marks' markers found (PDF may be image-only).")
        sys.exit(2)

    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    paper_id = add_paper(conn, args.board, tier, series, paper_no, calculator, year, args.url, args.markscheme_url)

    # Open again to extract snippets by page range
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        cur = conn.cursor()
        for (qno, marks, ps, pe) in ranges:
            # collect text across the page range
            pages_text = []
            for p in range(ps, pe+1):
                pages_text.append(pdf.pages[p-1].extract_text() or "")
            raw = " ".join(pages_text)    
            snippet = compact(clean_text(raw))
            topic = guess_topic(snippet)
            diff = 1 if marks <= 2 else (2 if marks <= 4 else 3)
            cur.execute("""
                INSERT OR REPLACE INTO questions
                (paper_id,qno,marks,page_start,page_end,topic,difficulty,text_snippet)
                VALUES(?,?,?,?,?,?,?,?)
            """, (paper_id, qno, marks, ps, pe, topic, diff, snippet))
        conn.commit()
    conn.close()
    print(f"Inserted {len(ranges)} questions (auto-tagged where possible).")

if __name__ == "__main__":
    main()
