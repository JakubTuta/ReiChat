# CLAUDE.md

Guidance for working in this repo. Local-only Ollama chatbot: Nuxt 3 (SPA, `ssr: false`) →
FastAPI (async, uvicorn) → Ollama containers spawned via the mounted Docker socket → PostgreSQL 16
+ pgvector through async SQLAlchemy and Alembic. No auth — single user, localhost-only by design.

## Engineering principles

1. **Never guess an API — verify with Context7.** Before writing against FastAPI, Starlette,
   Pydantic, SQLAlchemy (asyncio), Alembic, uvicorn, httpx, LangChain/`langchain-ollama`, the Docker
   SDK, uv, Nuxt, Vue, Vuetify, Pinia or pgvector, fetch
   current docs via Context7 (`resolve-library-id` → `query-docs`), even for things that look
   obvious. Where behavior can't be settled from docs, probe it live before shipping.
2. **Delete dead code rather than carry it.**
3. **Group logic by domain, one home per concern** — e.g. `backend/app/catalog/` (parser + registry
   client + sync), `backend/app/rag/` (extraction + chunking + embeddings + retrieval, kept apart
   from the chat socket that calls it), `backend/app/mcp_integration/` (MCP client),
   `backend/app/openai_compat/` (OpenAI-compatible request/response handling), Docker lifecycle
   (`containers/manager.py`) vs. Ollama HTTP calls (`llm/ollama_api.py`) vs. progress reporting
   (`realtime/progress.py`) kept separate, frontend `composables/` for WebSocket clients rather than
   `constants/`.
4. **Build for the roadmap, not just the fix.** WebSocket handlers must support cancellation, the
   catalog carries capability flags (tools/vision/embedding), the Ollama client must be reusable
   for embeddings. Don't hardcode assumptions that block these.
5. **Fail loudly, never silently.** No bare `except: pass`, no swallowed parse errors, no success
   returned on an empty or errored result. Every failure carries a message a non-technical user can
   act on.
6. **Types and tests travel with the code.** Type hints on new Python, no `any` in new TypeScript,
   a test with every fix.
7. **Small, reviewable commits, one concern each**, ordered so the tree is runnable at every step.
8. **This app is for people who are new to local AI.** Every user-facing flow (chat, model install,
   RAG, tools) needs a working empty state, a plain-language error message, and a guard against the
   obvious "something isn't ready yet" case (Docker down, model still pulling, container stopped) —
   never a silent no-op or a raw stack trace.

## Conventions

- WebSocket/API error payloads use lowercase `"error"` / `"status"` keys, consistently.
- Use `logger`, never `print`, in backend code.
- Alembic migrations are **committed** (`backend/migrations/versions/`). Generate one with
  `uv run alembic revision --autogenerate -m "..."`, read it before committing (autogenerate misses
  some changes), and commit it alongside the model change that produced it.
- Backend code is async end to end: never call blocking code (the Docker SDK, pypdf/docx parsing,
  `subprocess`, `requests`) on the event loop. The Docker SDK is imported only in
  `containers/manager.py`, which wraps it in `asyncio.to_thread`; other blocking work does the same.
- Database sessions: `SessionDep` never commits — the service function that owns the change commits
  on its success path, so a handled 4xx can't persist half of it. Don't hold a session open across
  a streamed reply. Relationships are `lazy="raise"`, so load what you need with
  `selectinload`/`joinedload` instead of relying on implicit loading.
- Commit messages and PR descriptions carry no AI attribution — no `Co-Authored-By` trailer, no
  "generated with" line. The only author is the person who owns the repo.
- Component/store naming and structure: see existing `stores/*.ts` (Pinia) and `components/*.vue`
  for the established patterns before adding new ones.
- Destructive or hard-to-reverse user actions (delete chat, delete document, delete MCP server,
  remove a model) always go through `ConfirmDialog.vue` — never a bare click-to-delete button.
- A backend dependency pin that "already works" in a stale local venv can still fail a clean
  `uv sync --locked` in the Docker image — always verify new/changed pins against a fresh
  `docker compose build`, not just the local environment.
- Comments carry WHY only, never WHAT. No decorative section dividers (`/* ── Label ── */`,
  `// ─── Label ───`, `<!-- ── Label ── -->`) above a self-evidently-named class/function/block. No
  comment that just repeats info already stated elsewhere (e.g. `# GET /foo/` on a route whose
  path is already in its decorator). No framework-boilerplate docstrings/comments (generated
  file headers, links to generic framework docs). A real WHY
  comment (bug history, race condition, protocol quirk, platform gotcha) stays — most existing
  comments in this repo are this kind and are correct as-is.

## Commands

```bash
# Backend (backend/) — Python is managed by uv; needs `docker compose up -d postgres`
uv sync                             # create .venv from uv.lock
uv run uvicorn app.asgi:app --reload  # dev server
uv run alembic upgrade head         # apply migrations
uv run pytest                       # tests
uv run ruff check .                 # lint
uv run ruff format .                # format
uv run mypy app                     # types
uv lock                             # after changing pyproject.toml

# Frontend (frontend/)
npm run dev                         # dev server
npm run test                        # vitest
npm run typecheck                   # vue-tsc
npx eslint .                        # lint
npx playwright test                 # e2e (needs the stack running)

# Full stack
docker-compose up -d
```

## Guardrails

- Never `delete(AIModel)` without a `where` (or add any catalog refresh path that deletes before a
  successful, validated re-scrape) — the foreign keys are `ondelete="CASCADE"`, so the database
  itself removes every `ChatHistory`/`ChatMessage` of those models instantly. Destroys user data.
- Never expose this app beyond `127.0.0.1` while `/var/run/docker.sock` is mounted into the
  backend container — that socket is equivalent to host root.
- Never claim a feature in `README.md` that isn't actually in the code.
