# GiveMeJobPlease (V1)

Personal, self-hosted Telegram bot for parsing, tracking, and ranking **EPFL IS-Academia** internship exports.

## Highlights
- **Zero-Scraping, Drop-to-Bot Workflow:** Hit `Ctrl+S` on IS-Academia (`Internships showroom (stu).htm` or `Registrations (stu).htm`) and drop the file into your Telegram bot chat.
- **Pipeline-Ready State Machine:** Tracks `new`, `seen`, `applied`, and `excluded` offers plus an independent `⭐ Star` flag. Eligible pool is always `Current Showroom - Applied - Excluded`, with one-tap toggles to revisit `seen`, `starred`, `applied`, or `excluded` offers.
- **In-Place Single-Message Telegram UX:** `/digest` and `/browse` render a compact ranked list in a single message edited in-place across **List → Offer Card → Paginated Full Description** (with `Star / Apply / Exclude` directly available in every view), marking `new` offers as `seen` on open or via `👀 Mark page seen` so pagination never skips offers.
- **Profile + CV Context:** Viewable `/profile` summary with a 5-step guided wizard (including a post-scoring min score threshold that never triggers paid re-scoring) plus `.pdf` / `.txt` / `.md` CV extraction. Each wizard step sends the question first and your **current value as a separate plain-text message** — long-press → Copy, edit, send back. `/cv` likewise shows the extracted text as a copyable plain-text preview. **Upload an anonymised CV only** — see [Data Handling & CV Privacy Policy](#data-handling--cv-privacy-policy).
- **Cancellable Scoring with Live Progress & Cost:** Long scoring runs show `60/418 · batch 4/28 · $0.0123 so far` (fed by OpenRouter's free per-response usage accounting) with a `🛑 Cancel scoring` button; `/cancel` also stops a run. Completed batches are checkpointed, so cancelling never re-bills finished work — `/digest` resumes where it stopped.
- **Dual Scoring Engine:**
  - **Mode B (Primary when connected):** Connect OpenRouter via secure headless **OAuth PKCE** (`/connect_openrouter` → `/auth <code>`) to score offers with DeepSeek, Qwen, or any OpenRouter model. Completely overrides local embeddings.
  - **Mode A (Default without API key):** Runs local multilingual ONNX embeddings (`fastembed`) on CPU with zero API keys and displays all eligible matches ranked by cosine similarity.
- **Observability & Versioning:** Every update gets a `trace_id`, secrets are automatically redacted from logs, SQLite migrations run via Alembic with pre-migration backups, and stale scores auto-invalidate via `scoring_fingerprint`.

## Quick Start (Docker — Recommended for Infomaniak VM)

1. Copy `.env.example` to `.env` and fill in `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USER_IDS`, and a generated `ENCRYPTION_KEY`:
   ```bash
   cp .env.example .env
   python3 -c "import os, base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
   ```
2. Launch with Docker Compose (zero inbound ports required — uses outbound Telegram long-polling):
   ```bash
   docker compose up -d --build
   docker compose logs -f
   ```

## Data Handling & CV Privacy Policy

**Do not upload your real CV.** Replace every identifying or sensitive value with mock
data before the file ever reaches the bot. Keep the substance (skills, technologies,
projects, languages, years of experience) — that is what the scorer actually reads.

| Field | Replace with |
| --- | --- |
| Full name | `John Doe` |
| Email | `john.doe@example.com` |
| Phone | `+41 00 000 00 00` |
| Street address | Region only, e.g. `Vaud`, `Zurich`, `Haute-Savoie` |
| Date of birth / nationality / marital status | Remove |
| Personal URLs, social handles | `example.com` |
| Employers / schools (optional) | Generic equivalents, e.g. `EPFL` → `Example University` |

The same rule applies to `/profile` free-text fields: they are copied verbatim into LLM
prompts and stored unencrypted, so keep them free of personal data.

### Where the CV actually goes

| Destination | What it sees | When |
| --- | --- | --- |
| Telegram servers | The uploaded file itself. Bot chats are **cloud chats, not end-to-end encrypted**, and anyone holding the bot token can download the file | Always |
| Local SQLite (`user_config.cv_text`) | Up to 20,000 chars of extracted text, stored as AES-256-GCM ciphertext (`enc:v1:` prefix, bound to your user id) | Always |
| Local SQLite backups (`data/backups/`) | Copies of the DB. Rows written before this encryption landed — and any backup taken before it — may still hold **plaintext** CV text until re-upload + expiry of old backups | Always |
| Local ONNX embedder (Mode A) | Nothing leaves the host | Default |
| OpenRouter + the selected upstream model provider (Mode B) | The first ~5,000 chars of CV text inside the scoring prompt, wrapped as untrusted data | Only while OpenRouter is connected |

Only the OpenRouter key and the CV text are encrypted; everything else in `./data` is
protected solely by filesystem permissions (`errors.jsonl` and DB backups are created
`0600`). Use `/digest local` to force offline scoring for a run, and
`/disconnect_openrouter` to wipe the stored key and return to Mode A permanently.

## Local Development (`uv`)

```bash
uv sync --extra dev
cp .env.example .env
# Edit .env, then run migrations and start:
uv run alembic upgrade head
uv run give-me-job-please
```

Run tests, linter, and type checker:
```bash
uv run pytest
uv run ruff check src tests
uv run mypy src
```

## Telegram Commands
- `/start` — Verify access & show quick help
- `/profile` — View current preferences & score threshold (`/profile edit` for the guided wizard, `/profile threshold <0-10|off>` for quick cutoff updates)
- `/cv` — Upload a `.pdf`, `.txt`, or `.md` CV (or drop it directly in chat) — **anonymised only**
- `/digest [3d|7d|14d|30d|all] [posted|ingested] [local]` — Score & view `new` offers; recency filtering runs **before** OpenRouter/local scoring so older offers do not consume LLM tokens (`posted` filters by IS-Academia posting date with fallback to upload date; `ingested` filters by bot upload date; `local` forces offline Mode A). *(Designed as an extensible pre-LLM filter stage for future rules like salary or company filters.)*
- `/browse [all|new|seen|starred|applied|excluded] [7d] [posted|ingested]` — Browse offers sorted by relevance with optional recency filtering
- `/connect_openrouter [limit] [expiration]` — Start headless OAuth PKCE flow
- `/auth <code>` — Exchange code for a scoped OpenRouter key (message is auto-deleted)
- `/model [search_or_id]` — Search OpenRouter catalog or select active model
- `/disconnect_openrouter` — Wipe stored key and revert to Mode A
- `/export` — Receive enriched `internships.csv` + `internships_descriptions.json`
- `/cancel` — Stop an active scoring run (completed batches stay saved) or exit the setup wizard
- `/status` — Show versions, active scorer, DB counts, OpenRouter key links, last-run token/cost summary, and key spend-today / remaining limit
- `/logs` — Show the last 5 traced error entries

### Cancelling a scoring run
Tap `🛑 Cancel scoring` on the progress message or send `/cancel`. Notes:
- Every completed batch is already persisted; the next `/digest` re-scores **only** the
  offers that never got a score, so cancelled work is not billed twice.
- A batch already sent to OpenRouter may still be billed server-side even if you cancel
  while waiting for its response — cancellation cannot un-send an in-flight request.
- In local Mode A, an in-flight ONNX embedding chunk finishes in its worker thread
  (it cannot be killed); the run stops right after that chunk.
- The cost display uses OpenRouter's automatic per-response `usage.cost`; some providers
  omit it, in which case the bot shows `cost n/a` rather than guessing.
