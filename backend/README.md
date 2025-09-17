Adaptive Quiz backend

Endpoints

- GET /health
- GET /topics
- GET /papers
- GET /questions?paper_id=<id>
- POST /attempt  (body: student_id, topic, difficulty, correct)
- GET /recommendation?student_id=<id>&k=3&policy=baseline|cf|logreg|hybrid
- GET /recommend_questions?student_id=<id>&k=5&policy=baseline|cf|logreg|hybrid
- POST /train  (trigger model retrain)

Notes

- `recommend_questions` uses TF-IDF over `questions.text_snippet` when available to rank question-level items; falls back to topic mapping otherwise.
- A/B logging is written to `ab_log` table; use `backend/ab_analysis.py` to inspect results.

Quick run (PowerShell)

```powershell
python -m pip install -r backend/requirements.txt
python backend/synth_generate.py --students 40 --topics 8 --length 120 --db app.db
python backend/offline_eval.py --k 3 --test_frac 0.2 --n_boot 100 --db app.db
uvicorn backend.main:app --reload
```
