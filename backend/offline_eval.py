# backend/offline_eval.py
import os, math, argparse, sqlite3, sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

# Optional plotting (safe if matplotlib missing or on headless Windows)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Import your backend
sys.path.append(os.path.dirname(__file__))
import main as backend  # uses your train_models, recommend_* and active_topics

# ---------- helpers ----------
def temporal_split(df, test_frac=0.2):
    parts = []
    for sid, g in df.sort_values("ts").groupby("student_id"):
        n = len(g)
        k = max(1, int(math.ceil(n * test_frac)))
        tr, te = g.iloc[:-k], g.iloc[-k:]
        parts.append((sid, tr.copy(), te.copy()))
    return parts

def true_weak_topics(test_df, top_m=3):
    perf = test_df.groupby("topic")["correct"].mean().sort_values()
    return set(perf.index[:min(top_m, len(perf))])

def rank_metrics(rec_list, relevant_set, k=3):
    topk = rec_list[:k]
    hits = [1 if t in relevant_set else 0 for t in topk]
    prec = sum(hits)/max(1,k)
    rec = sum(hits)/max(1,len(relevant_set))
    dcg = sum(h/np.log2(i+2) for i,h in enumerate(hits))
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant_set), k)))
    ndcg = dcg/(idcg or 1.0)
    return prec, rec, ndcg

def ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
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

def bootstrap_mean_ci(values, n_boot=400, alpha=0.05, rng=None):
    arr = np.array(values, dtype=float)
    if rng is None: rng = np.random.default_rng(0)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return arr.mean(), lo, hi

def build_lr_features_for_row(lr_meta, train_hist_by_st_topic, row):
    # Match backend LR features: [prev_mean, difficulty, days_since_topic] + one-hot(topic)
    sid, topic = row["student_id"], row["topic"]
    diff = float(row["difficulty"])
    ts = pd.to_datetime(row["ts"])
    hist = train_hist_by_st_topic.get((sid, topic), [])
    if hist:
        prev_mean = np.mean([h[1] for h in hist])
        last_seen = max([h[0] for h in hist])
        days_since = int((ts - last_seen).days)
    else:
        prev_mean = 0.5
        days_since = 60
    xnum = np.array([prev_mean, diff, float(np.clip(days_since,0,90))], dtype=float)
    topics_n = lr_meta["topics_n"]; topic_index = lr_meta["topic_index"]
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

# ---------- main evaluate ----------
def evaluate(db="app.db", k=3, test_frac=0.2, n_boot=400, seed=7):
    rng = np.random.default_rng(seed)
    os.makedirs("reports", exist_ok=True)

    # Load full dataset
    conn = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT student_id, topic, difficulty, correct, ts FROM attempts ORDER BY ts", conn)
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.tz_convert(None)

    # Temporal splits per student
    splits = temporal_split(df, test_frac=test_frac)
    df_train_all = pd.concat([tr for _, tr, _ in splits], axis=0)

    # Train models on TRAIN only
    prepare_train_db(df_train_all, db=db)
    backend.train_models()

    # Precompute TRAIN histories for LR feature building
    train_hist = {}
    for sid, tr, _ in splits:
        for _, r in tr.iterrows():
            key = (r["student_id"], r["topic"])
            train_hist.setdefault(key, []).append((pd.to_datetime(r["ts"]), int(r["correct"])))

    # Choose policies (include UCB if present)
    policies = ["baseline", "cf", "logreg", "hybrid"]
    if hasattr(backend, "recommend_lr_ucb"):
        policies.insert(3, "logreg_ucb")

    # ---- Ranking evaluation ----
    rank_rows = []
    for sid, tr, te in splits:
        relevant = true_weak_topics(te, top_m=k)
        for pol in policies:
            recs = backend.get_recommendation(student_id=sid, k=k, policy=pol).next_topics
            p, r, n = rank_metrics(recs, relevant, k=k)
            rank_rows.append(dict(student_id=sid, policy=pol, prec=p, rec=r, ndcg=n))

    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv("reports/metrics_rank.csv", index=False)

    # Bootstrap CIs
    agg = []
    for pol in policies:
        sub = rank_df[rank_df["policy"] == pol]
        for m in ["prec","rec","ndcg"]:
            mean, lo, hi = bootstrap_mean_ci(sub[m].values, n_boot=n_boot, rng=rng)
            agg.append(dict(policy=pol, metric=m, mean=mean, ci_lo=lo, ci_hi=hi))
    ci_df = pd.DataFrame(agg)
    ci_df.to_csv("reports/metrics_rank_ci.csv", index=False)

    # ---- Prediction evaluation (LR) ----
    pred_df = None
    if backend.MODELS.get("lr") is not None:
        lr_meta = backend.MODELS["lr"]
        lr = lr_meta["model"]
        y_true, y_prob = [], []
        for _, _, te in splits:
            X = np.vstack([build_lr_features_for_row(lr_meta, train_hist, r) for _, r in te.iterrows()])
            probs = lr.predict_proba(X)[:,1]
            y_prob.extend(probs.tolist())
            y_true.extend(te["correct"].astype(int).tolist())
        y_true = np.array(y_true, dtype=int)
        y_prob = np.clip(np.array(y_prob, dtype=float), 1e-6, 1-1e-6)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float("nan")
        ll = log_loss(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        cal = ece(y_true, y_prob, n_bins=10)
        pred_df = pd.DataFrame([dict(model="logreg", auc=auc, logloss=ll, brier=brier, ece=cal)])
        pred_df.to_csv("reports/metrics_pred.csv", index=False)
    # return everything for optional plotting
    return dict(rank_df=rank_df, ci_df=ci_df, pred_df=pred_df)

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--db", type=str, default="app.db")
    args = ap.parse_args()

    out = evaluate(db=args.db, k=args.k, test_frac=args.test_frac, n_boot=args.n_boot)

    # ---- Plots (optional) ----
    if plt is not None:
        ci_df = out["ci_df"]
        # keep a consistent order for bars
        order = ["baseline","cf","logreg","logreg_ucb","hybrid"]
        for metric, fname in [("prec","plot_rank_precision.png"), ("ndcg","plot_rank_ndcg.png")]:
            sub = ci_df[ci_df["metric"] == metric].copy()
            sub["policy"] = pd.Categorical(sub["policy"], categories=[p for p in order if p in sub["policy"].unique()], ordered=True)
            sub = sub.sort_values("policy")
            x = np.arange(len(sub))
            y = sub["mean"].values
            err = np.vstack([y - sub["ci_lo"].values, sub["ci_hi"].values - y])
            plt.figure(figsize=(7,4))
            plt.bar(x, y)
            plt.errorbar(x, y, yerr=err, fmt="none", capsize=4)
            plt.xticks(x, sub["policy"].tolist())
            plt.ylabel(f"{metric.upper()}@{args.k}")
            plt.title(f"{metric.upper()} (95% CI)")
            plt.tight_layout()
            plt.savefig(os.path.join("reports", fname), dpi=150)
            plt.close()

        # Calibration placeholder (uses summary if raw not kept)
        pred_df = out["pred_df"]
        if pred_df is not None:
            ece_val = float(pred_df.loc[0, "ece"]) if "ece" in pred_df.columns else None
            plt.figure(figsize=(5,5))
            plt.plot([0,1],[0,1])
            plt.xlim(0,1); plt.ylim(0,1)
            plt.xlabel("Predicted probability"); plt.ylabel("Empirical accuracy")
            title = "Calibration (LR)"
            if ece_val is not None:
                title += f" — ECE={ece_val:.3f}"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join("reports","plot_calibration.png"), dpi=150)
            plt.close()
