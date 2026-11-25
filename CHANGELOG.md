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
