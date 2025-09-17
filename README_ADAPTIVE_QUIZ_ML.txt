Adaptive Quiz ML Pack

Files:
- ml_pipeline_adaptive_quiz.ipynb   Notebook with CF, LR, Hybrid, CV, plots
- ml_pipeline_adaptive_quiz.py      Standalone script version
- adaptive_quiz_interactions.csv    Data snapshot (auto generated if your results.csv missing)
- ml_metrics_folds.csv              Per-fold metrics from 5-fold CV
- ml_metrics_summary.csv            Summary stats
- plot_logloss_by_model.png         Bar chart of log loss
- plot_auc_by_model.png             Bar chart of ROC AUC

How to run the script:
1) Put results.csv in the same folder if you have real data. Columns needed: student_id, topic, correct. Optional: timestamp, difficulty.
2) Run: python ml_pipeline_adaptive_quiz.py
3) Outputs will be saved in the current folder.

How to run the notebook:
Open ml_pipeline_adaptive_quiz.ipynb in Jupyter or VS Code and run all cells.
