# backend/make_report_tables.py
import os
import pandas as pd

OUT_DIR = os.path.join("reports", "tables")
os.makedirs(OUT_DIR, exist_ok=True)

def save_all_formats(df: pd.DataFrame, stem: str):
    md = os.path.join(OUT_DIR, f"{stem}.md")
    tex = os.path.join(OUT_DIR, f"{stem}.tex")
    csv = os.path.join(OUT_DIR, f"{stem}_Word.csv")
    # Markdown
    try:
        df.to_markdown(md, index=False)
    except Exception:
        # old pandas without to_markdown: write CSV as fallback
        df.to_csv(md + ".csv", index=False)
    # LaTeX
    with open(tex, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False))
    # Word-friendly CSV
    df.to_csv(csv, index=False)
    print(f"Wrote:\n  {md}\n  {tex}\n  {csv}")

# ---------- Table 1: Ranking metrics with 95% CI ----------
ci_path = os.path.join("reports", "metrics_rank_ci.csv")
if not os.path.exists(ci_path):
    raise SystemExit(f"Missing {ci_path}. Run offline_eval.py first.")

ci = pd.read_csv(ci_path)  # expects: policy, metric, mean, ci_lo, ci_hi

def fmt_row(g):
    out = {}
    for m in ["prec", "rec", "ndcg"]:
        row = g[g["metric"] == m].iloc[0]
        out[m] = f'{row["mean"]:.3f} [{row["ci_lo"]:.3f}–{row["ci_hi"]:.3f}]'
    return pd.Series(out)

tab1 = ci.groupby("policy", as_index=False).apply(fmt_row)
tab1 = tab1.rename(columns={"policy": "Policy",
                            "prec": "Prec@3", "rec": "Rec@3", "ndcg": "nDCG@3"})
save_all_formats(tab1, "Table1_Ranking")

# ---------- Table 2: Prediction/calibration (LogReg) ----------
pred_path = os.path.join("reports", "metrics_pred.csv")
if not os.path.exists(pred_path):
    raise SystemExit(f"Missing {pred_path}. Run offline_eval.py first.")

pred = pd.read_csv(pred_path)  # expects: model, auc, logloss, brier, ece
tab2 = pred.copy()
for c in ["auc", "logloss", "brier", "ece"]:
    if c in tab2.columns:
        tab2[c] = tab2[c].map(lambda x: f"{x:.4f}")
tab2 = tab2.rename(columns={"model": "Model",
                            "auc": "AUC", "logloss": "LogLoss",
                            "brier": "Brier", "ece": "ECE"})
save_all_formats(tab2, "Table2_Prediction")

# ---------- Optional appendix tables ----------
rank_path = os.path.join("reports", "metrics_rank.csv")
if os.path.exists(rank_path):
    df = pd.read_csv(rank_path)  # per-student
    # A: per-student winner by nDCG
    winners = (df.sort_values(["student_id", "ndcg"], ascending=[True, False])
                 .drop_duplicates("student_id"))
    counts = winners["policy"].value_counts().reset_index()
    counts.columns = ["Policy", "#Students won"]
    save_all_formats(counts, "Appendix_WinsByPolicy")

    # B: top students where Hybrid beats Baseline (nDCG gap)
    pvt = df.pivot_table(index="student_id", columns="policy", values="ndcg", aggfunc="mean")
    pvt["hybrid_minus_baseline"] = pvt.get("hybrid") - pvt.get("baseline")
    top = (pvt.sort_values("hybrid_minus_baseline", ascending=False)
               .head(10)
               .reset_index())
    save_all_formats(top, "Appendix_TopHybridVsBaseline")

print("Done.")
