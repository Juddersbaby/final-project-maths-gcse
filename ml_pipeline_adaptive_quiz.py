#!/usr/bin/env python3
"""
Adaptive Quiz ML Pipeline
- Baseline heuristic
- CF via Truncated SVD
- Logistic Regression
- Hybrid blend
Performs 5-fold CV and saves metrics and plots.
"""

import os, math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, log_loss, roc_auc_score
import matplotlib.pyplot as plt

DATA_IN = "results.csv"  # if not found, will use adaptive_quiz_interactions.csv or synthesise
DATA_FALLBACK = "adaptive_quiz_interactions.csv"

def load_or_synthesise(path=DATA_IN, n_users=120, n_topics=14, seed=42):
    rng = np.random.default_rng(seed)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'timestamp' not in df.columns:
            base = datetime(2025, 3, 1); df['timestamp'] = [base + timedelta(days=int(i/5)) for i in range(len(df))]
        if 'difficulty' not in df.columns:
            topic_levels = {t: 1 + (i % 3) for i, t in enumerate(sorted(df['topic'].unique()))}
            df['difficulty'] = df['topic'].map(topic_levels).astype(int)
        if 'student_id' not in df.columns and 'user' in df.columns:
            df['student_id'] = df['user']
        return df
    if os.path.exists(DATA_FALLBACK):
        return pd.read_csv(DATA_FALLBACK, parse_dates=['timestamp'])
    topics = ["Number","Algebra","Ratio & Proportion","Geometry","Trigonometry","Probability","Statistics","Sequences","Graphs","Transformations","Vectors","Equations","Inequalities","Functions"][:n_topics]
    user_ids = np.arange(1000, 1000+n_users)
    user_global = rng.normal(0.0, 0.6, size=n_users)
    user_topic_offsets = rng.normal(0.0, 0.5, size=(n_users, len(topics)))
    rows = []; start = datetime(2025, 3, 1)
    for u_idx, uid in enumerate(user_ids):
        n_attempts = rng.integers(60, 121)
        last_time = start + timedelta(days=int(rng.integers(0, 30)))
        for _ in range(n_attempts):
            t_idx = int(rng.integers(0, len(topics))); topic = topics[t_idx]
            difficulty = int(rng.integers(1, 4))
            recency_days = float(rng.integers(0, 60))
            ability = user_global[u_idx] + 0.8*user_topic_offsets[u_idx, t_idx]
            spacing_boost = 0.02 * max(0, 30 - recency_days) / 30.0
            logit = ability - 0.6*(difficulty-2) + spacing_boost + rng.normal(0, 0.5)
            p = 1/(1+math.exp(-logit)); correct = 1 if rng.random() < p else 0
            last_time += timedelta(days=int(rng.integers(0, 4)))
            rows.append({"student_id": uid,"topic": topic,"difficulty": difficulty,"timestamp": last_time,"correct": correct})
    return pd.DataFrame(rows)

def engineer(df):
    df = df.sort_values(['student_id','timestamp']).copy()
    df['topic_id'] = df['topic'].astype('category').cat.codes
    df['prev_time'] = df.groupby(['student_id','topic'])['timestamp'].shift(1)
    df['days_since_topic'] = (df['timestamp'] - df['prev_time']).dt.days.fillna(60).astype(int).clip(0,90)
    grp = df.groupby(['student_id','topic'])['correct']
    cumsum = grp.cumsum() - df['correct']
    cumcnt = grp.cumcount()
    prev_mean = np.where(cumcnt>0, cumsum / cumcnt, np.nan)
    df['prev_correct_mean'] = pd.Series(prev_mean, index=df.index).fillna(0.5)
    def baseline_predict_prob(row):
        base = row['prev_correct_mean']
        adj = base - 0.1*(row['difficulty']-2) - 0.001*(row['days_since_topic']-30)
        return float(np.clip(adj, 0.05, 0.95))
    df['p_baseline'] = df.apply(baseline_predict_prob, axis=1)
    return df

def cf_train_predict(train_df, test_df, n_components=8):
    mat = train_df.groupby(['student_id','topic'])['correct'].mean().unstack(fill_value=np.nan)
    users = mat.index; topics = mat.columns
    topic_means = mat.mean(axis=0).fillna(train_df['correct'].mean())
    mat_filled = mat.fillna(topic_means)
    mat_centered = mat_filled - topic_means
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    U = svd.fit_transform(mat_centered); S = svd.singular_values_; Vt = svd.components_
    recon = (U @ np.diag(S) @ Vt) + topic_means.values
    recon_df = pd.DataFrame(recon, index=users, columns=topics)
    def lookup(row):
        u, t = row['student_id'], row['topic']
        if (u in recon_df.index) and (t in recon_df.columns):
            return float(recon_df.loc[u, t])
        return float(topic_means.get(t, train_df['correct'].mean()))
    return test_df.apply(lookup, axis=1).clip(0.05,0.95).values

def build_lr_features(df_in, topics_n=None):
    X_num = df_in[['prev_correct_mean','difficulty','days_since_topic']].to_numpy(dtype=float)
    if topics_n is None:
        topics_n = int(df_in['topic_id'].max()+1)
    topic_oh = np.zeros((len(df_in), topics_n))
    topic_oh[np.arange(len(df_in)), df_in['topic_id'].astype(int).values] = 1.0
    return np.hstack([X_num, topic_oh])

def evaluate_models(df, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []; fold=0; topics_n = int(df['topic_id'].max()+1)
    for tr, te in kf.split(df):
        fold += 1
        tr_df, te_df = df.iloc[tr].copy(), df.iloc[te].copy()
        y = te_df['correct'].values
        p_base = te_df['p_baseline'].values
        p_cf = cf_train_predict(tr_df, te_df, n_components=8)
        X_tr = build_lr_features(tr_df, topics_n); X_te = build_lr_features(te_df, topics_n)
        lr = LogisticRegression(max_iter=500); lr.fit(X_tr, tr_df['correct'].values)
        p_lr = lr.predict_proba(X_te)[:,1]
        best_w, best_ll = None, 1e9
        tv = int(0.85*len(tr_df))
        X_tr2, X_va = X_tr[:tv], X_tr[tv:]; y_va = tr_df['correct'].values[tv:]
        val_cf = cf_train_predict(tr_df.iloc[:tv], tr_df.iloc[tv:], n_components=8)
        val_lr = lr.predict_proba(X_va)[:,1]
        for w in np.linspace(0,1,11):
            p_val = (w*val_cf + (1-w)*val_lr).clip(1e-3,1-1e-3)
            ll = log_loss(y_va, p_val)
            if ll < best_ll: best_ll, best_w = ll, w
        p_h = (best_w*p_cf + (1-best_w)*p_lr).clip(1e-3,1-1e-3)
        for name, p in [('Baseline',p_base), ('CF-SVD',p_cf), ('LogReg',p_lr), (f'Hybrid(w={best_w:.2f})',p_h)]:
            acc = accuracy_score(y, p>0.5)
            pr, rc, f1, _ = precision_recall_fscore_support(y, (p>0.5).astype(int), average='binary', zero_division=0)
            try: auc = roc_auc_score(y, p)
            except: auc = np.nan
            ll = log_loss(y, p)
            rows.append({"fold":fold,"model":name,"accuracy":acc,"precision":pr,"recall":rc,"f1":f1,"roc_auc":auc,"log_loss":ll})
    return pd.DataFrame(rows)

def main():
    df = load_or_synthesise()
    df.to_csv("adaptive_quiz_interactions.csv", index=False)
    df = engineer(df)
    metrics_df = evaluate_models(df, n_splits=5)
    summary = metrics_df.groupby('model').agg(['mean','std']).reset_index()
    summary.columns = ['model'] + [f'{a}_{b}' for a,b in summary.columns.tolist()[1:]]
    metrics_df.to_csv("ml_metrics_folds.csv", index=False)
    summary.to_csv("ml_metrics_summary.csv", index=False)
    # Plots
    import matplotlib; matplotlib.use('Agg')
    plt.figure()
    metrics_df.groupby('model')['log_loss'].mean().plot(kind='bar')
    plt.title('Mean Log Loss by Model (lower is better)')
    plt.ylabel('Log Loss'); plt.xlabel('Model'); plt.tight_layout()
    plt.savefig("plot_logloss_by_model.png"); plt.close()
    plt.figure()
    metrics_df.groupby('model')['roc_auc'].mean().plot(kind='bar')
    plt.title('Mean ROC AUC by Model (higher is better)')
    plt.ylabel('ROC AUC'); plt.xlabel('Model'); plt.tight_layout()
    plt.savefig("plot_auc_by_model.png"); plt.close()
    print("Saved: ml_metrics_folds.csv, ml_metrics_summary.csv, plot_logloss_by_model.png, plot_auc_by_model.png")

if __name__ == "__main__":
    main()
