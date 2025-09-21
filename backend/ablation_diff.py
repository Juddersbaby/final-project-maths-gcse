# backend/ablation_diff.py
import sys, pandas as pd, os

def load(path):
    df = pd.read_csv(path)
    # keep only what we need
    return df[["policy","metric","mean","ci_lo","ci_hi"]].rename(
        columns={"mean":"mean","ci_lo":"lo","ci_hi":"hi"}
    )

def main(baseline_ci, ablated_ci, out_csv, policy="logreg"):
    base = load(baseline_ci)
    abl  = load(ablated_ci)
    mrg = base.merge(abl, on=["policy","metric"], suffixes=("_base","_abl"))
    # filter to a single policy if you like (logreg for Ablation A)
    mrg = mrg[mrg["policy"]==policy].copy()
    mrg["delta"] = mrg["mean_abl"] - mrg["mean_base"]
    tab = (mrg[["metric","mean_base","mean_abl","delta","lo_base","hi_base","lo_abl","hi_abl"]]
           .sort_values("metric"))
    os.makedirs("reports\\tables", exist_ok=True)
    tab.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    print(tab)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python backend\\ablation_diff.py <baseline_ci.csv> <ablated_ci.csv> <out.csv> [policy]")
        sys.exit(1)
    baseline_ci, ablated_ci, out_csv = sys.argv[1:4]
    policy = sys.argv[4] if len(sys.argv) > 4 else "logreg"
    main(baseline_ci, ablated_ci, out_csv, policy)
