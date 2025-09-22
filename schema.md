# Database schema (SQLite)

This document summarises the database schema as defined in `backend/main.py` and `backend/ingest_paper.py`.

- Database file: `app.db`
- Engine: SQLite

## Tables

### attempts
- id INTEGER PRIMARY KEY AUTOINCREMENT
- student_id TEXT NOT NULL
- topic TEXT NOT NULL
- difficulty INTEGER NOT NULL
- correct INTEGER NOT NULL
- ts TEXT NOT NULL

Notes: Stores each logged attempt by a student.

### papers
- id INTEGER PRIMARY KEY AUTOINCREMENT
- board TEXT
- tier TEXT
- series TEXT
- paper_no INTEGER
- calculator INTEGER
- year INTEGER
- pdf_url TEXT NOT NULL
- markscheme_url TEXT
- UNIQUE(board, tier, series, paper_no)

Notes: Metadata about past papers and links to PDFs and mark schemes.

### questions
- id INTEGER PRIMARY KEY AUTOINCREMENT
- paper_id INTEGER NOT NULL REFERENCES papers(id)
- qno INTEGER NOT NULL
- marks INTEGER
- page_start INTEGER NOT NULL
- page_end INTEGER NOT NULL
- topic TEXT
- difficulty INTEGER
- open_url TEXT (added via migration)
- text_snippet TEXT (added via migration/ingestion)
- UNIQUE(paper_id, qno)

Notes: Page range allows deep-linking into PDFs; `open_url` is backfilled to `pdf_url#page=<page_start>` when available. `text_snippet` holds extracted text for TF‑IDF ranking.

### ab_log
- id INTEGER PRIMARY KEY AUTOINCREMENT
- student_id TEXT
- policy TEXT
- k INTEGER
- payload TEXT
- ts TEXT

Notes: Records which recommendation policy served what payload for A/B analysis.

### classes
- id INTEGER PRIMARY KEY AUTOINCREMENT
- name TEXT NOT NULL UNIQUE
- created_at TEXT
- curriculum TEXT (JSON string of up to 25 weeks)

Notes: Created by `migrate_schema()`; curriculum stored as JSON array of week topics.

### students
- id INTEGER PRIMARY KEY AUTOINCREMENT
- student_id TEXT NOT NULL UNIQUE
- name TEXT
- class_id INTEGER REFERENCES classes(id)
- created_at TEXT

Indexes:
- idx_students_class_id ON students(class_id)

Notes: Students optionally linked to a class.

## Migrations and backfills
- `migrate_schema()` ensures `classes` and `students` exist, adds `curriculum` to `classes` if missing, and adds `open_url` and `text_snippet` to `questions` if missing.
- Backfill: sets `questions.open_url` to the paper `pdf_url` plus `#page=<page_start>` where absent.
- `backend/ingest_paper.py::ensure_schema()` also ensures `papers` and `questions` exist and adds `text_snippet` to `questions` if missing.

## Relations
- questions.paper_id → papers.id (many questions per paper)
- students.class_id → classes.id (many students per class)
- attempts.student_id is a free-text foreign key to students.student_id (not enforced by SQLite)
- ab_log.student_id mirrors attempts.student_id (not enforced)

## Usage notes
- The API writes to these tables via `backend/main.py` endpoints.
- TF‑IDF over `questions.text_snippet` is built in-memory and not persisted.
- All timestamps are stored as TEXT (ISO‑8601 recommended).

## Source
- Core DDL: `backend/main.py` (init_db, migrate_schema)
- Ingestion DDL: `backend/ingest_paper.py` (ensure_schema)