# Changelog

All notable changes to **GiveMeJobPlease** are documented in this file.
This project follows [Semantic Versioning](https://semver.org/).

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
