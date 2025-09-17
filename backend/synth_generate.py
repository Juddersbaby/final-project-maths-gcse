# backend/synth_generate.py
import numpy as np, pandas as pd, argparse, time, sqlite3, datetime as dt, itertools, random

TOPICS = ["Number","Algebra","Ratio & Proportion","Geometry","Trigonometry","Probability",
          "Statistics","Sequences","Graphs","Transformations","Vectors","Equations","Inequalities","Functions"]

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def generate(S=60, T=8, L=120, seed=42):
    rng = np.random.default_rng(seed)
    topics = TOPICS[:T]
    d = 3  # latent dim
    A = rng.normal(0, 1, size=(S, d))                 # student abilities
    B = rng.normal(0, 1, size=(T, d))                 # topic loadings
    delta = rng.normal(0.0, 0.5, size=T)              # topic bias (difficulty)
    # timeline per student
    rows = []
    start = dt.datetime(2024,1,1)
    for s in range(S):
        last_seen = {t: None for t in topics}
        a = A[s].copy()
        for step in range(L):
            # pick a topic to attempt uniformly at random to generate logs (off-policy)
            t_idx = rng.integers(0, T)
            t = topics[t_idx]
            # recency decay (if not seen recently, slight decay)
            if last_seen[t] is not None:
                days = (step - last_seen[t])
                a = a - 0.002*days*(a*0)              # mild global decay; keep simple
            x = 1.2*np.dot(a, B[t_idx]) - delta[t_idx]
            p = sigmoid(x)
            y = rng.random() < p
            # learning update: small boost on topic component if attempted
            a = a + 0.06 * B[t_idx] * (0.5 + 0.5*int(y))  # larger if correct
            last_seen[t] = step
            # map marks->difficulty buckets to match your API
            diff = 1 + int((delta[t_idx] > 0.3)) + int((delta[t_idx] > 0.9))
            ts = (start + dt.timedelta(days=step)).isoformat()
            rows.append(dict(student_id=f"S{s:03d}", topic=t, difficulty=diff, correct=int(y), ts=ts))
    return pd.DataFrame(rows)

def write_sqlite(df, db="app.db"):
    conn = sqlite3.connect(db); cur = conn.cursor()
    cur.execute("DELETE FROM attempts")  # wipe for fresh run
    conn.commit()
    df.to_sql("attempts", conn, if_exists="append", index=False)
    conn.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=60)
    ap.add_argument("--topics",   type=int, default=8)
    ap.add_argument("--length",   type=int, default=120)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--db",       type=str, default="app.db")
    args = ap.parse_args()
    df = generate(args.students, args.topics, args.length, args.seed)
    write_sqlite(df, db=args.db)
    print(df.head(), "\nInserted:", len(df))
