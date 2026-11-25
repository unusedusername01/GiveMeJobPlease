# Changelog

All notable changes to **GiveMeJobPlease** are documented in this file.
This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Security
- Dockerfile pins `python:3.12-slim` and `ghcr.io/astral-sh/uv:0.13.10` by image digest
  and runs `uv sync --locked`, so the image installs exactly the reviewed lockfile set
  and fails if `uv.lock` is stale.
- Added `.dockerignore` (keeps `.env`, `data/`, `*.db*`, `.git/`, `.venv/`, caches and
  `tests/` out of the build context).
- `cv_text` is now encrypted at rest with AES-256-GCM under an `enc:v1:` prefix, bound to
  the owning user id; legacy plaintext rows remain readable and are encrypted on the next
  save.
- CSV exports neutralise spreadsheet formula injection (`=`, `+`, `-`, `@`, and control
  characters at cell start).
- Log redaction now catches Bot API URLs (`/bot<token>`) and bare `ENCRYPTION_KEY` values,
  and traverses tuples; `errors.jsonl` and pre-migration DB backups are created `0600`.
- Scoring prompts delimit candidate/offer data as untrusted and forbid following embedded
  instructions; each OpenRouter batch is checkpointed, and a per-user lock prevents
  duplicate concurrent (billable) scoring runs.
- Container hardening: `no-new-privileges`, `cap_drop: ALL`, read-only root filesystem,
  `tmpfs` for `/tmp` and `~/.cache`, `pids_limit: 128`.
### Changed
- PDF CV parsing rejects documents over 20 pages, and parsers run off the event loop.
- `TELEGRAM_ALLOWED_USER_IDS` must be non-empty (fail fast at startup).
### Added
- `/profile` settings summary view with inline actions, `/profile edit`, `/profile threshold <0-10|off>`, and Step 5/5 (`min_score_threshold`, persisted via reversible Alembic migration `0002_add_min_score_threshold`) excluded from `profile_hash` so cutoff edits never trigger paid re-scoring.
- Pre-scoring recency filtering for `/digest` and `/browse` (`3d`, `7d`, `14d`, `30d`, `all`) running **before** OpenRouter/local scoring, defaulting to IS-Academia posting date (`creation_date`, falling back to `first_seen_at`) with CLI and inline toggles to upload date (`first_seen_at`), structured as an extensible pre-LLM filter pipeline.
- Single-message in-place Telegram UX for `/digest` and `/browse`: compact ranked list with numbered `[1]..[N]` drill-down, contextual `Prev / Next / Back to list` card navigation, paginated in-place full descriptions with direct `Star / Apply / Exclude` actions, and `seen` tracking on open or via `👀 Mark page seen` (eliminating `new`-view page skipping).
- README "Data Handling & CV Privacy Policy": upload an anonymised CV (mock name, email,
  phone, address) and a table of every destination the CV text can reach.

## [0.1.0] - 2026-11-25
### Added
- Deterministic EPFL IS-Academia `.htm` parsers (`Internships showroom (stu).htm` and `Registrations (stu).htm`) with automatic file-type and encoding detection (`windows-1252` / `utf-8`).
- Offer lifecycle state machine (`new`, `seen`, `applied`, `excluded`) with orthogonal `is_starred` bookmarking and `Eligible Pool = Showroom - Applied - Excluded`.
- Interactive `/profile` FSM setup wizard and `/cv` parser (`.pdf`, `.txt`, `.md`).
- Mode A (Zero-Key Default): Local multilingual ONNX embeddings via `fastembed` (`paraphrase-multilingual-MiniLM-L12-v2`) with SQLite vector caching.
- Mode B (Primary when connected): Headless OpenRouter OAuth PKCE (`/connect_openrouter` + `/auth <code>`), AES-256-GCM encrypted key storage, live `/model` catalog search, and batch structured LLM scoring overriding Mode A.
- Structured JSON logging (`structlog`) with per-update `trace_id`, automatic secret redaction, centralized error handler, and `/logs` owner inspection.
- Versioned SQLite schema via Alembic with automatic pre-migration SQLite backup on container startup.
- Production Docker + `docker-compose.yml` deployment setup for CPU-only VMs (Infomaniak Public Cloud).
