from fastapi import FastAPI, HTTPException, Query
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import sqlite3, datetime, random
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import Query
import json
import logging, traceback
logger = logging.getLogger("uvicorn.error")


DB_PATH = "app.db"
TOPICS = [
    "Number","Algebra","Ratio & Proportion","Geometry","Trigonometry","Probability",
    "Statistics","Sequences","Graphs","Transformations","Vectors","Equations","Inequalities","Functions"
]

def active_topics():
    """
    Topics that exist in the current dataset/models.
    Prefer model topics if trained; otherwise read from DB; otherwise use the full TOPICS constant.
    """
    try:
        # If CF is trained, its recon_df columns are the trained topic set
        cf = MODELS.get("cf")
        if cf and isinstance(cf.get("recon", None), pd.DataFrame):
            cols = list(cf["recon"].columns)
            if cols: return cols
        # If LR is trained, its topic_index keys form the set
        lr = MODELS.get("lr")
        if lr and lr.get("topic_index"):
            keys = list(lr["topic_index"].keys())
            if keys: return keys
    except Exception:
        pass

    # Fallback: read from attempts table
    try:
        df = load_df()
        if not df.empty:
            return sorted(df["topic"].unique().tolist())
    except Exception:
        pass

    # Last resort: full constant
    return TOPICS



def recommend_ucb(user: str, k: int):
    # per-topic Beta(1+success, 1+fail) from this student's history
    df = load_df()
    if df.empty: return random.sample(TOPICS, k=min(k, len(TOPICS)))
    g = df[df["student_id"]==user].groupby("topic")["correct"].agg(["sum","count"])
    scores = {}
    for t in TOPICS:
        s = g.loc[t]["sum"] if t in g.index else 0
        c = g.loc[t]["count"] if t in g.index else 0
        # UCB from Beta posterior mean + uncertainty bonus
        mean = (1+s)/(2+c)
        bonus = np.sqrt(2*np.log(1+g["count"].sum()+1) / (1+c))
        scores[t] = mean + 0.2*bonus
    # Sort descending by score (higher is better)
    ranked = sorted(TOPICS, key=lambda x: scores[x], reverse=True)
    return ranked[:k]


def init_db():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    # attempts table (already in your app)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            difficulty INTEGER NOT NULL,
            correct INTEGER NOT NULL,
            ts TEXT NOT NULL
        )
    """)
    # NEW: papers and questions metadata (links + page ranges only)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            board TEXT,
            tier TEXT,
            series TEXT,
            paper_no INTEGER,
            calculator INTEGER,
            year INTEGER,
            pdf_url TEXT NOT NULL,
            markscheme_url TEXT,
            UNIQUE(board,tier,series,paper_no)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL,
            qno INTEGER NOT NULL,
            marks INTEGER,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            topic TEXT,
            difficulty INTEGER,
            FOREIGN KEY(paper_id) REFERENCES papers(id),
            UNIQUE(paper_id, qno)
        )
    """)
    # A/B logging table: stores which policy was served and timestamp
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ab_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            policy TEXT,
            k INTEGER,
            payload TEXT,
            ts TEXT
        )
    """)
    conn.commit(); conn.close()
init_db()

def migrate_schema():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    # Ensure classes and students tables exist
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL UNIQUE,
            name TEXT,
            class_id INTEGER,
            created_at TEXT,
            FOREIGN KEY(class_id) REFERENCES classes(id)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_students_class_id ON students(class_id)")
    # Ensure questions has open_url and text_snippet
    cols = [r[1] for r in cur.execute("PRAGMA table_info(questions)").fetchall()]
    if "open_url" not in cols:
        cur.execute("ALTER TABLE questions ADD COLUMN open_url TEXT")
    if "text_snippet" not in cols:
        cur.execute("ALTER TABLE questions ADD COLUMN text_snippet TEXT")
    # Best-effort backfill open_url
    try:
        cur.execute("""
            UPDATE questions
            SET open_url = (SELECT pdf_url FROM papers WHERE papers.id = questions.paper_id)
                           || '#page=' || COALESCE(page_start,1)
            WHERE (open_url IS NULL OR open_url = '')
        """)
    except Exception:
        pass
    conn.commit(); conn.close()

init_db()
migrate_schema()

def build_tfidf_questions():
    """Build and cache TF-IDF matrix for all questions (text_snippet).
    Stores vectorizer and matrix in MODELS['tfidf'].
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, topic, text_snippet FROM questions")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        MODELS['tfidf'] = None
        return
    ids, topics, texts = zip(*rows)
    texts = [t if t is not None else "" for t in texts]
    vec = TfidfVectorizer(max_features=4096)
    X = vec.fit_transform(texts)
    MODELS['tfidf'] = {"vectorizer": vec, "X": X, "ids": list(ids), "topics": list(topics)}


def ensure_tfidf():
    if MODELS.get('tfidf') is None:
        try:
            build_tfidf_questions()
        except Exception:
            MODELS['tfidf'] = None


app = FastAPI(title="Adaptive Quiz API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000"],
    allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

class AttemptIn(BaseModel):
    student_id: str
    topic: str
    difficulty: int = Field(ge=1, le=3)
    correct: int = Field(ge=0, le=1)
    timestamp: Optional[str] = None

class RecommendationOut(BaseModel):
    policy: str
    student_id: str
    next_topics: List[str]

class ClassIn(BaseModel):
    name: str = Field(min_length=1)

class StudentIn(BaseModel):
    student_id: str = Field(min_length=1)
    name: Optional[str] = None

@app.get("/health")
def health():
    return {"ok": True, "version": "0.3.0"}

@app.get("/topics")
def get_topics():
    return [{"topic": t} for t in active_topics()]

# ---------- Papers & Questions endpoints ----------
def _get_paper_url(paper_id: int) -> str:
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("SELECT pdf_url FROM papers WHERE id=?", (paper_id,))
    row = cur.fetchone(); conn.close()
    return row[0] if row else ""

@app.get("/papers")
def list_papers(board: str = "Edexcel", tier: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cur = conn.cursor()
    if tier:
        cur.execute("SELECT * FROM papers WHERE board=? AND tier=? ORDER BY year DESC, paper_no", (board, tier))
    else:
        cur.execute("SELECT * FROM papers WHERE board=? ORDER BY year DESC, paper_no", (board,))
    out = [dict(r) for r in cur.fetchall()]
    conn.close()
    return out

@app.get("/questions")
def list_questions(paper_id: int = Query(..., description="papers.id to fetch")):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cur = conn.cursor()
    cur.execute("SELECT * FROM questions WHERE paper_id=? ORDER BY qno", (paper_id,))
    out = [dict(r) for r in cur.fetchall()]
    conn.close()
    base = _get_paper_url(paper_id)
    for r in out:
        r["open_url"] = f"{base}#page={r['page_start']}"
    return out

# ---------- Attempts & Recommenders  ----------
@app.post("/attempt")
def post_attempt(a: AttemptIn):
    if a.topic not in TOPICS:
        raise HTTPException(status_code=400, detail="Unknown topic")
    ts = a.timestamp or datetime.datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("""
        INSERT INTO attempts(student_id, topic, difficulty, correct, ts)
        VALUES (?, ?, ?, ?, ?)
    """, (a.student_id.strip(), a.topic, int(a.difficulty), int(a.correct), ts))
    conn.commit(); conn.close()
    MODELS["dirty"] = True
    return {"ok": True}

# ---------- Classes & Students (lightweight CRUD) ----------
@app.get("/classes")
def list_classes():
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cur = conn.cursor()
    cur.execute("SELECT c.id, c.name, COALESCE(cnt.cnt,0) AS student_count FROM classes c LEFT JOIN (SELECT class_id, COUNT(1) AS cnt FROM students GROUP BY class_id) cnt ON c.id = cnt.class_id ORDER BY c.name")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

@app.post("/classes")
def create_class(payload: ClassIn):
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    try:
        ts = datetime.datetime.utcnow().isoformat()
        cur.execute("INSERT INTO classes(name, created_at) VALUES(?, ?)", (payload.name.strip(), ts))
        cid = cur.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=409, detail="Class already exists")
    conn.close()
    return {"id": cid, "name": payload.name}

@app.delete("/classes/{class_id}")
def delete_class(class_id: int):
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    # Detach students first (keep their identity and attempts)
    cur.execute("UPDATE students SET class_id = NULL WHERE class_id = ?", (class_id,))
    cur.execute("DELETE FROM classes WHERE id = ?", (class_id,))
    conn.commit(); conn.close()
    return {"ok": True}

@app.get("/classes/{class_id}/students")
def list_class_students(class_id: int):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cur = conn.cursor()
    cur.execute("SELECT id, student_id, name, class_id, created_at FROM students WHERE class_id = ? ORDER BY student_id", (class_id,))
    out = [dict(r) for r in cur.fetchall()]
    conn.close(); return out

@app.post("/classes/{class_id}/students")
def add_student_to_class(class_id: int, payload: StudentIn):
    sid = payload.student_id.strip()
    if not sid:
        raise HTTPException(status_code=400, detail="student_id required")
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    ts = datetime.datetime.utcnow().isoformat()
    # Upsert-like: try insert; if exists, update class_id and name
    try:
        cur.execute("INSERT INTO students(student_id, name, class_id, created_at) VALUES(?, ?, ?, ?)", (sid, payload.name, class_id, ts))
    except sqlite3.IntegrityError:
        cur.execute("UPDATE students SET class_id = ?, name = COALESCE(?, name) WHERE student_id = ?", (class_id, payload.name, sid))
    conn.commit(); conn.close()
    return {"ok": True, "student_id": sid, "class_id": class_id}

@app.delete("/classes/{class_id}/students/{student_id}")
def remove_student_from_class(class_id: int, student_id: str):
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("UPDATE students SET class_id = NULL WHERE class_id = ? AND student_id = ?", (class_id, student_id))
    conn.commit(); conn.close()
    return {"ok": True}

@app.post("/students/{student_id}/upload_csv")
async def upload_student_csv(student_id: str, file: UploadFile = File(...)):
    # Expect CSV with columns: topic, difficulty, correct, ts(optional)
    content = await file.read()
    try:
        from io import StringIO
        buf = StringIO(content.decode('utf-8', errors='ignore'))
        df = pd.read_csv(buf)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    required = {"topic", "difficulty", "correct"}
    if not required.issubset(set(map(str.lower, df.columns))):
        # try case-insensitive mapping
        cols_map = {c.lower(): c for c in df.columns}
        if not required.issubset(cols_map.keys()):
            raise HTTPException(status_code=400, detail="CSV must have columns: topic, difficulty, correct, [ts]")
        df = df.rename(columns={cols_map.get("topic"): "topic", cols_map.get("difficulty"): "difficulty", cols_map.get("correct"): "correct", cols_map.get("ts"): "ts" if cols_map.get("ts") else None})

    # Normalize and validate
    df["topic"] = df["topic"].astype(str)
    df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce").fillna(2).clip(1,3).astype(int)
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).clip(0,1).astype(int)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    else:
        df["ts"] = pd.NaT

    # Filter to known topics and cap rows
    df = df[df["topic"].isin(TOPICS)].head(5000)
    if df.empty:
        return {"ok": True, "inserted": 0}

    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    now_iso = datetime.datetime.utcnow().isoformat()
    rows = []
    for _, r in df.iterrows():
        ts = r["ts"].isoformat() if pd.notna(r["ts"]) else now_iso
        rows.append((student_id.strip(), r["topic"], int(r["difficulty"]), int(r["correct"]), ts))
    cur.executemany("INSERT INTO attempts(student_id, topic, difficulty, correct, ts) VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit(); conn.close()
    MODELS["dirty"] = True
    return {"ok": True, "inserted": len(rows)}

def load_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT student_id, topic, difficulty, correct, ts FROM attempts ORDER BY ts ASC",
        conn
    )
    conn.close()
    if df.empty:
        return df
    # Parse ANY incoming datetimes as UTC then drop tz to make them tz-naive
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.tz_convert(None)
    return df


def build_features(df: pd.DataFrame):
    df = df.sort_values(["student_id","ts"]).copy()
    grp = df.groupby(["student_id","topic"])["correct"]
    cumsum = grp.cumsum() - df["correct"]
    cnt = grp.cumcount()
    df["prev_mean"] = np.where(cnt>0, cumsum/ cnt, 0.5)
    prev_time = df.groupby(["student_id","topic"])["ts"].shift(1)
    days = (df["ts"] - prev_time).dt.days.fillna(60).clip(0, 90).astype(int)
    df["days_since_topic"] = days
    df["topic_id"] = df["topic"].astype("category").cat.codes
    return df

def train_models():
    df = load_df()
    if df.empty:
        MODELS["cf"] = None; MODELS["lr"] = None; MODELS["dirty"] = False; MODELS["meta"] = {}
        return
    df = build_features(df)
    mat = df.groupby(["student_id","topic"])["correct"].mean().unstack(fill_value=np.nan)
    topic_means = mat.mean(axis=0).fillna(df["correct"].mean())
    mat_filled = mat.fillna(topic_means)
    mat_centered = mat_filled - topic_means
    svd = TruncatedSVD(n_components=min(8, min(mat_centered.shape)-1), random_state=0)
    U = svd.fit_transform(mat_centered); S = svd.singular_values_; Vt = svd.components_
    recon = (U @ np.diag(S) @ Vt) + topic_means.values
    recon_df = pd.DataFrame(recon, index=mat.index, columns=mat.columns).clip(0.05, 0.95)
    topics_n = int(df["topic_id"].max() + 1)
    X_num = df[["prev_mean","difficulty","days_since_topic"]].to_numpy(float)
    topic_oh = np.zeros((len(df), topics_n)); topic_oh[np.arange(len(df)), df["topic_id"].values] = 1.0
    X = np.hstack([X_num, topic_oh]); y = df["correct"].values
    lr = LogisticRegression(max_iter=500, solver="lbfgs"); lr.fit(X, y)
    MODELS["cf"] = {"recon": recon_df, "topic_means": topic_means}
    MODELS["lr"] = {"model": lr, "topics_n": topics_n, "topic_index": dict(zip(df["topic"].astype("category").cat.categories, range(topics_n)))}
    MODELS["dirty"] = False
    MODELS["meta"] = {"users": mat.index.tolist(), "topics": list(mat.columns)}

def topic_days_since(user: str):
    df = load_df()
    cand = active_topics()
    out = {t: 60 for t in cand}
    if df.empty:
        return out
    dfu = df[df["student_id"] == user]
    if dfu.empty:
        return out
    last = dfu.groupby("topic")["ts"].max().to_dict()
    # Ensure tz-naive 'now' to match df["ts"]
    now = pd.Timestamp.utcnow().tz_localize(None)
    for t in cand:
        if t in last and pd.notna(last[t]):
            out[t] = int((now - last[t]).days)
    return out



def recommend_cf(user: str, k: int):
    m = MODELS.get("cf")
    cand = active_topics()
    if not m:
        return random.sample(cand, k=min(k, len(cand)))
    recon_df, topic_means = m["recon"], m["topic_means"]
    if user in recon_df.index:
        probs = recon_df.loc[user].to_dict()
    else:
        probs = {t: float(topic_means.get(t, 0.6)) for t in cand}
    days = topic_days_since(user)
    scored = sorted(cand, key=lambda t: (1.0 - probs.get(t, 0.6)) + 0.002*days.get(t,60), reverse=True)
    return scored[:k]


def recommend_lr(user: str, k: int, difficulty: int = 2):
    m = MODELS.get("lr")
    cand = active_topics()
    if not m:
        return random.sample(cand, k=min(k, len(cand)))
    lr = m["model"]; topics_n = m["topics_n"]; topic_index = m["topic_index"]
    df = load_df()
    prev = df[df["student_id"] == user].groupby("topic")["correct"].mean().to_dict()
    days = topic_days_since(user)
    rows = []
    for t in cand:
        pm = prev.get(t, 0.5)
        ds = days.get(t, 60)
        xnum = np.array([pm, float(difficulty), float(ds)], dtype=float)
        oh = np.zeros(topics_n)
        idx = topic_index.get(t, None)
        if idx is not None:
            oh[idx] = 1.0
        rows.append((t, np.concatenate([xnum, oh])))
    X = np.vstack([r[1] for r in rows])
    probs = lr.predict_proba(X)[:,1]
    topic_prob = {rows[i][0]: float(probs[i]) for i in range(len(rows))}
    scored = sorted(cand, key=lambda t: (1.0 - topic_prob[t]) + 0.002*days.get(t,60), reverse=True)
    return scored[:k]


def recommend_hybrid(user: str, k: int):
    a = recommend_cf(user, len(TOPICS)); b = recommend_lr(user, len(TOPICS))
    rank_a = {t:i for i,t in enumerate(a)}; rank_b = {t:i for i,t in enumerate(b)}
    scored = sorted(TOPICS, key=lambda t: -(1.0/(1+rank_a.get(t,99)) + 1.0/(1+rank_b.get(t,99))))
    return scored[:k]

def safe_cf(user: str, k: int):
    try:
        out = recommend_cf(user, k)
        if not out: out = random.sample(TOPICS, k=min(k, len(TOPICS)))
        return [t for t in out if t in TOPICS][:k]
    except Exception:
        logger.exception("CF recommendation failed.")
        return random.sample(TOPICS, k=min(k, len(TOPICS)))

def safe_lr(user: str, k: int):
    try:
        out = recommend_lr(user, k)
        if not out: out = random.sample(TOPICS, k=min(k, len(TOPICS)))
        return [t for t in out if t in TOPICS][:k]
    except Exception:
        logger.exception("LogReg recommendation failed.")
        return random.sample(TOPICS, k=min(k, len(TOPICS)))

def safe_hybrid(user: str, k: int):
    try:
        a = recommend_cf(user, len(TOPICS)) or []
        b = recommend_lr(user, len(TOPICS)) or []
        if not a and not b:
            return random.sample(TOPICS, k=min(k, len(TOPICS)))
        if not a: return b[:k]
        if not b: return a[:k]
        rank_a = {t:i for i,t in enumerate(a)}
        rank_b = {t:i for i,t in enumerate(b)}
        scored = sorted(TOPICS, key=lambda t: -(1/(1+rank_a.get(t,99)) + 1/(1+rank_b.get(t,99))))
        out = [t for t in scored if t in TOPICS][:k]
        if len(out) < k:
            pad = [t for t in TOPICS if t not in out]
            out = (out + pad)[:k]
        return out
    except Exception:
        logger.exception("Hybrid recommendation failed; falling back to CF.")
        return safe_cf(user, k)


MODELS = {"cf": None, "lr": None, "dirty": True, "meta": {}}
def ensure_models():
    if MODELS.get("dirty", True) or (MODELS["cf"] is None and MODELS["lr"] is None):
        train_models()

# Pre-build TF-IDF question index at startup so recommend_questions can respond fast
try:
    ensure_tfidf()
except Exception:
    MODELS['tfidf'] = None

@app.get("/recommendation", response_model=RecommendationOut)
def get_recommendation(student_id: str, k: int = 3, policy: str = "baseline"):
    policy = policy.lower().strip()

    # If a model policy is requested, ensure models exist
    if policy in ("cf", "logreg", "hybrid"):
        try:
            ensure_models()
        except Exception:
            logger.exception("ensure_models() failed; downgrading to baseline.")
            policy = "baseline"

    # Active topic universe (ALWAYS define this before using)
    cand = active_topics()
    if not cand:
        cand = TOPICS

    # Fetch student's history (safe)
    try:
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("""
            SELECT topic, difficulty, correct, ts
            FROM attempts
            WHERE student_id = ?
            ORDER BY ts ASC
        """, (student_id,))
        rows = cur.fetchall()
        conn.close()
    except Exception:
        logger.exception("DB fetch in /recommendation failed; using empty history.")
        rows = []

    # Policy routing (safe wrappers for model policies)
    if policy == "cf":
        topics = safe_cf(student_id, k)
    elif policy in ("logreg", "lr"):
        topics = safe_lr(student_id, k)
    elif policy == "hybrid":
        topics = safe_hybrid(student_id, k)
    else:
        # Baseline: weakness + staleness over the ACTIVE set
        if not rows:
            topics = random.sample(cand, k=min(k, len(cand)))
        else:
            from collections import defaultdict
            stats = defaultdict(lambda: {"sum":0,"cnt":0,"last":None,"prev":0.5})
            for topic, difficulty, correct, ts in rows:
                d = stats[topic]
                d["prev"] = d["sum"]/d["cnt"] if d["cnt"]>0 else 0.5
                d["sum"] += int(correct); d["cnt"] += 1; d["last"] = ts

            now = datetime.datetime.utcnow()
            def score(t):
                d = stats.get(t, {"prev":0.5,"last":None})
                if d["last"]:
                    try:
                        last = datetime.datetime.fromisoformat(d["last"])
                    except Exception:
                        last = now - datetime.timedelta(days=60)
                    days = (now - last).days
                else:
                    days = 60
                return (1.0 - d["prev"]) + 0.002 * days

            topics = sorted(cand, key=lambda t: score(t), reverse=True)[:k]

    # Final sanitation: ensure exactly k topics from cand
    topics = [t for t in topics if t in cand][:k]
    if len(topics) < k:
        topics = (topics + [t for t in cand if t not in topics])[:k]

    return RecommendationOut(policy=policy, student_id=student_id, next_topics=topics)


@app.get("/recommend_questions")
def recommend_questions(student_id: str = Query(...), k: int = Query(5, ge=1, le=20), policy: str = "logreg"):
    # 1) get topic recs (always works because it’s the same logic as /recommendation)
    try:
        topics = get_recommendation(student_id=student_id, k=min(max(k,3), len(TOPICS)), policy=policy).next_topics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get topics: {e}")

    # load question table as DataFrame
    def get_questions_df():
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT id, paper_id, qno, topic, marks, difficulty, page_start, page_end, open_url, text_snippet FROM questions", conn)
        conn.close()
        if df.empty:
            return df
        df["text"] = df["text_snippet"].fillna("").astype(str).str.replace(r"\s+"," ", regex=True).str.strip()
        return df

    qdf = get_questions_df()
    if qdf.empty:
        # Return empty list rather than 404 so callers/tests can handle gracefully
        return {"policy": policy, "student_id": student_id, "k": k, "topics_used": topics, "items": [], "questions": []}

    cand_df = qdf[qdf["topic"].isin(topics)].copy()
    if cand_df.empty:
        cand_df = qdf.copy()

    # 2) try TF-IDF ranking if available, else ignore any TF-IDF errors and fall back
    ranked_ids = []
    try:
        ensure_tfidf()
        tf = MODELS.get('tfidf')
        if tf:
            from sklearn.metrics.pairwise import cosine_similarity
            X = tf["X"]; tf_ids = tf["ids"]; tf_topics = tf["topics"]
            # Build topic centroids in TF-IDF space
            cents = {}
            topics_arr = np.array(tf_topics, dtype=object)
            for t in pd.Series(topics_arr).dropna().unique():
                idx = np.where(topics_arr == t)[0]
                if len(idx) > 0:
                    cents[t] = X[idx].mean(axis=0)
            # User vector = sum of chosen topic centroids
            user_vec = None; cnt = 0
            for t in topics:
                c = cents.get(t)
                if c is not None:
                    user_vec = c if user_vec is None else (user_vec + c)
                    cnt += 1
            if user_vec is not None and cnt > 0:
                sims = cosine_similarity(user_vec, X)
                sims = np.asarray(sims).ravel()
                cand_ids = set(int(x) for x in cand_df["id"].values.tolist())
                for idx in np.argsort(sims)[::-1]:
                    qid = int(tf_ids[idx])
                    if qid in cand_ids:
                        ranked_ids.append(qid)
                        if len(ranked_ids) >= k*3:
                            break
    except Exception:
        ranked_ids = []  # ignore TF-IDF issues and fall back

    # 3) assemble output without duplicates
    seen, out_rows = set(), []

    def push_rows(frame):
        # get papers url map once
        conn = sqlite3.connect(DB_PATH)
        paper_url = pd.read_sql_query("SELECT id, pdf_url FROM papers", conn).set_index("id")["pdf_url"].to_dict()
        conn.close()
        for _, r in frame.iterrows():
            rid = int(r["id"])
            if rid in seen:
                continue
            # compute open_url if missing
            ou = r.get("open_url")
            if not ou:
                base = paper_url.get(int(r["paper_id"])) if pd.notna(r["paper_id"]) else None
                if base:
                    ou = f"{base}#page={int(r['page_start']) if pd.notna(r['page_start']) else 1}"

            out_rows.append({
                "id": rid,
                "paper_id": int(r["paper_id"]) if pd.notna(r["paper_id"]) else None,
                "qno": int(r["qno"]) if pd.notna(r["qno"]) else None,
                "topic": r["topic"],
                "marks": int(r["marks"]) if pd.notna(r["marks"]) else None,
                "difficulty": int(r["difficulty"]) if pd.notna(r["difficulty"]) else None,
                "page_start": int(r["page_start"]) if pd.notna(r["page_start"]) else None,
                "page_end": int(r["page_end"]) if pd.notna(r["page_end"]) else None,
                "open_url": ou,
            })
            seen.add(rid)
            if len(out_rows) >= k:
                break


    # 3a) use TF-IDF ordering if we have it
    if ranked_ids:
        ordered = cand_df.set_index("id").loc[[i for i in ranked_ids if i in cand_df["id"].values]].reset_index()
        push_rows(ordered)

    # 3b) fill from candidate topics randomly (consistent seed for reproducibility)
    if len(out_rows) < k:
        push_rows(cand_df.sample(frac=1.0, random_state=0))

    # 3c) pad from any topic if still short
    if len(out_rows) < k:
        rest = qdf[~qdf["id"].isin(seen)].sample(frac=1.0, random_state=1)
        push_rows(rest)

    # 4) best-effort A/B log (won’t crash on errors)
    try:
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("""
          CREATE TABLE IF NOT EXISTS ab_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            student_id TEXT NOT NULL,
            policy TEXT NOT NULL,
            item_type TEXT NOT NULL,
            item_ids TEXT NOT NULL
          )
        """)
        now = datetime.datetime.utcnow().isoformat()
        ids_str = ",".join(str(r["id"]) for r in out_rows)
        cur.execute("INSERT INTO ab_log(ts, student_id, policy, item_type, item_ids) VALUES (?, ?, ?, ?, ?)",
                    (now, student_id, policy, "question", ids_str))
        conn.commit(); conn.close()
    except Exception:
        pass

    return {"policy": policy, "student_id": student_id, "k": k, "topics_used": topics, "items": out_rows, "questions": out_rows}


@app.post("/train")
def train_now():
    train_models()
    return {"ok": True, "meta": MODELS.get("meta", {})}
