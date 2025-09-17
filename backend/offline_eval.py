import os, math, argparse, sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

# Import your backend code directly
import sys
sys.path.append(os.path.dirname(__file__))
import main as backend  # uses backend.load_df, train_models, MODELS, etc.

TOPICS = backend.TOPICS

def temporal_split(df, test_frac=0.2):
    parts = []
    for sid, g in df.sort_values("ts").groupby("student_id"):
        n = len(g)
        k = max(1, int(math.ceil(n * test_frac)))
        tr, te = g.iloc[:-k], g.iloc[-k:]
        parts.append((sid, tr.copy(), te.copy()))
    return parts

def true_weak_topics(test_df, top_m=3):
    # topics with lowest correctness in the test window
    perf = test_df.groupby("topic")["correct"].mean().sort_values()
    return set(perf.index[:min(top_m, len(perf))])

def rank_metrics(rec_list, relevant_set, k=3):
    topk = rec_list[:k]
    hits = [1 if t in relevant_set else 0 for t in topk]
    prec = sum(hits)/max(1,k)
    rec = sum(hits)/max(1,len(relevant_set))
    # nDCG
    dcg = sum(h/np.log2(i+2) for i,h in enumerate(hits))
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant_set), k)))
    ndcg = dcg/(idcg or 1.0)
    return prec, rec, ndcg

def ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.digitize(y_prob, bins) - 1
    e = 0.0; N = len(y_true)
    for b in range(n_bins):
        mask = (idx == b)
        if not np.any(mask): continue
        conf = y_prob[mask].mean()
        acc  = y_true[mask].mean()
        e += abs(conf - acc) * (mask.sum()/N)
    return e

def build_lr_features_for_row(lr_meta, train_hist_by_student_topic, last_ts_by_student_topic, row):
    """Match backend's LR features: [prev_mean, difficulty, days_since_topic] + one-hot(topic)."""
    sid, topic, difficulty, ts = row["student_id"], row["topic"], float(row["difficulty"]), pd.to_datetime(row["ts"])
    # prev_mean from TRAIN history only
    hist = train_hist_by_student_topic.get((sid, topic), [])
    if hist:
        prev_mean = np.mean([h[1] for h in hist])
        last_seen = max([h[0] for h in hist])
        days_since = int((ts - last_seen).days)
    else:
        prev_mean = 0.5
        days_since = 60
    xnum = np.array([prev_mean, difficulty, float(np.clip(days_since, 0, 90))], dtype=float)
    topics_n = lr_meta["topics_n"]
    topic_index = lr_meta["topic_index"]
    oh = np.zeros(topics_n, dtype=float)
    idx = topic_index.get(topic, None)
    if idx is not None:
        oh[idx] = 1.0
    return np.concatenate([xnum, oh])

def prepare_train_db(df_train, db="app.db"):
    conn = sqlite3.connect(db); cur = conn.cursor()
    cur.execute("DELETE FROM attempts"); conn.commit()
    df_train.to_sql("attempts", conn, if_exists="append", index=False)
    conn.close()

def bootstrap_mean_ci(values, n_boot=500, alpha=0.05, rng=None):
    arr = np.array(values, dtype=float)
    if rng is None: rng = np.random.default_rng(0)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return arr.mean(), lo, hi

def evaluate(db="app.db", k=3, test_frac=0.2, n_boot=400, seed=7):
    rng = np.random.default_rng(seed)

    # Load full synthetic dataset
    conn = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT student_id, topic, difficulty, correct, ts FROM attempts ORDER BY ts", conn)
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"])

    # Temporal splits per student
    splits = temporal_split(df, test_frac=test_frac)
    df_train_all = pd.concat([tr for _, tr, _ in splits], axis=0)

    # Train models on TRAIN portion only
    prepare_train_db(df_train_all, db=db)
    backend.train_models()

    # Precompute TRAIN histories for LR feature building
    train_hist = {}
    last_ts = {}
    for sid, tr, _ in splits:
        for _, r in tr.iterrows():
            key = (r["student_id"], r["topic"])
            train_hist.setdefault(key, []).append((pd.to_datetime(r["ts"]), int(r["correct"])))
            last_ts[key] = pd.to_datetime(r["ts"])

    # ---- Ranking evaluation ----
    rank_rows = []
    policies = ["baseline", "cf", "logreg", "hybrid"]
    for sid, tr, te in splits:
        rel = true_weak_topics(te, top_m=k)
        for pol in policies:
            recs = backend.get_recommendation(student_id=sid, k=k, policy=pol).next_topics
            p, r, n = rank_metrics(recs, rel, k=k)
            rank_rows.append(dict(student_id=sid, policy=pol, prec=p, rec=r, ndcg=n))

    rank_df = pd.DataFrame(rank_rows)
    os.makedirs("reports", exist_ok=True)
    rank_df.to_csv("reports/metrics_rank.csv", index=False)

    # Bootstrap CIs per policy
    agg = []
    for pol in policies:
        sub = rank_df[rank_df["policy"] == pol]
        for m in ["prec", "rec", "ndcg"]:
            mean, lo, hi = bootstrap_mean_ci(sub[m].values, n_boot=n_boot, rng=rng)
            agg.append(dict(policy=pol, metric=m, mean=mean, ci_lo=lo, ci_hi=hi))
    rank_summary = pd.DataFrame(agg).pivot(index="policy", columns="metric", values="mean")
    print("\nRanking (mean):")
    print(rank_summary.round(3))
    pd.DataFrame(agg).to_csv("reports/metrics_rank_ci.csv", index=False)

    # ---- Prediction evaluation (LR only) ----
    lr_meta = backend.MODELS["lr"]
    if lr_meta is not None:
        y_true, y_prob = [], []
        lr = lr_meta["model"]
        for _, _, te in splits:
            # predict probability for each test attempt (using TRAIN-only features)
            X = np.vstack([build_lr_features_for_row(lr_meta, train_hist, last_ts, r) for _, r in te.iterrows()])
            probs = lr.predict_proba(X)[:,1]
            y_prob.extend(probs.tolist())
            y_true.extend(te["correct"].astype(int).tolist())

        # metrics
        y_true = np.array(y_true, dtype=int)
        y_prob = np.clip(np.array(y_prob, dtype=float), 1e-6, 1-1e-6)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float("nan")
        ll = log_loss(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        cal = ece(y_true, y_prob, n_bins=10)
        pred_row = pd.DataFrame([dict(model="logreg", auc=auc, logloss=ll, brier=brier, ece=cal)])
        pred_row.to_csv("reports/metrics_pred.csv", index=False)
        print("\nPrediction (LR):")
        print(pred_row.round(4))
    else:
        print("\nPrediction: LR model unavailable (skipped).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--db", type=str, default="app.db")
    args = ap.parse_args()
    evaluate(db=args.db, k=args.k, test_frac=args.test_frac, n_boot=args.n_boot)
