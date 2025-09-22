# Personalised Maths GCSE Practice (Teacher Dashboard)

Welcome! This repo is my GCSE Final Project: a small system that recommends maths topics to practise and shows questions from past papers. I built a FastAPI backend and a clean React dashboard so a teacher (or I!) can try the idea quickly.

## What this does
- Manage classes and students (simple roster tools)
- Log quick attempts per topic (correct/incorrect, difficulty)
- Get 3 topic recommendations per student
- Ingest past papers (MathsGenie links) and browse their questions
- Open the PDF at the right page; link to mark schemes when available
- Plan a 25‑week curriculum per class (JSON stored in the DB)

## How it works (quick overview)
- Backend: FastAPI + SQLite (single `app.db`). Models include a simple collaborative filter, a logistic regression, and safe fallbacks. There’s also a TF‑IDF index over question snippets for question‑level ranking.
- Frontend: React (Create React App) with Tailwind styles. It’s intentionally simple and fast.
- Data: Tables are documented in [`schema.md`](./schema.md). An ER diagram (SVG/PNG) lives in `reports/`.

## Screens at a glance
- Classes: create/delete classes, see student counts, open a class
- Students: add/remove students, upload CSV attempts, jump to detail view
- Control Panel: pick a student (filterable), pick topic + difficulty, log an attempt
- Curriculum: edit a 25‑week plan (Week 1 required; add weeks up to 25; repeats allowed)
- Student Detail: recent results + 3 recommendations; quick refresh
- Papers: paste a MathsGenie paper URL to ingest; list papers, toggle questions, open PDFs; mark scheme links when available

## Run it (Windows + PowerShell)
Backend (one‑time):
- Install Python 3.10+ and pip.
- From the project root:
  - `python -m pip install -r backend/requirements.txt`
  - Optional: generate synthetic data for a quick demo: `python backend/synth_generate.py --students 40 --topics 8 --length 120 --db app.db`
  - Train/evaluate offline (optional): `python backend/offline_eval.py --k 3 --test_frac 0.2 --n_boot 100 --db app.db`
- Start the API:
  - `python -m uvicorn backend.main:app --reload --port 8000`

Frontend:
- In a new PowerShell window, from the project root:
  - `cd frontend`
  - `npm install --no-audit --no-fund`
  - `npm start`
- The app opens on http://localhost:3000 and talks to the API at http://127.0.0.1:8000 (set in `frontend/.env`).

Tip: If port 3000 is busy, CRA will offer another port — accept it.

## API quick tour
- GET `/health` → check server is alive
- GET `/topics` → list known topics
- POST `/attempt` → log `{ student_id, topic, difficulty, correct }`
- GET `/recommendation?student_id=U1&k=3&policy=baseline|cf|logreg|hybrid` → 3 topics
- GET `/recommend_questions?student_id=U1&k=5` → question‑level suggestions if text available
- Classes/Students CRUD, curriculum endpoints, CSV uploads
- Papers/Questions: list and ingest from URLs

Explore everything via the built‑in OpenAPI docs at http://127.0.0.1:8000/docs.

## Database
- File: `app.db` (SQLite, easy to inspect)
- Schema: see [`schema.md`](./schema.md)
- ER diagram: `reports/er_diagram.svg` (and `.png`)

## Sample data & CSVs
- A sample CSV exists at `frontend/public/sample_student_results.csv`.
- Upload via the Students page or Student Detail view.

## Design choices (short and honest)
- SQLite to reduce operational risk and keep everything local.
- FastAPI for clear, typed endpoints.
- Simple models that are easy to explain in a report.
- TF‑IDF at question level to bridge from topics to concrete items.
- A/B logging table as a hook for future experiments.

## Limitations (future work)
- No authentication (deliberate for a local school demo).
- Content tagging is light; better NLP/metadata would help.
- Models are simple baselines — good enough for demonstration, not production.
- UI is plain by design; focus is on clarity and speed.

## Folder map
```
backend/   # FastAPI, models, ingestion, eval
frontend/  # React dashboard
reports/   # metrics, tables, ER diagram
public/    # static index.html for simple hosting
schema.md  # schema summary and ER diagram (Mermaid)
app.db     # SQLite database (gitignored by default)
```

## Screenshots (optional)
Drop images in `reports/` (e.g., `reports/ui_overview.png`) and link them here for your submission.

## Credits
- Past paper source parsing via simple rules for MathsGenie links
- Libraries: FastAPI, scikit‑learn, pandas, pdfplumber, requests, React, Tailwind

If you try this and spot a bug or have ideas, open an issue or PR. Thanks for reading!
