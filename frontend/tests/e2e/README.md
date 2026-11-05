# End-to-end tests

`smoke.spec.ts` covers the onboarding path that's safe to run in CI: every
top-level page loads, nav works, and the bundled seed catalog (Phase 0.1)
renders without any network call. It does **not** pull a real model —
a real pull is multi-gigabyte and can take minutes, which doesn't belong in
a CI job that runs on every push.

## Running locally

The app is a multi-container stack (Postgres + FastAPI + Nuxt), not
a single dev server, so Playwright does not manage a `webServer` here.
Bring the stack up yourself first:

```bash
docker compose up -d
npx playwright test
```

## What's intentionally not covered here

The plan's full verification path — "seed catalog -> create container ->
first answer" — needs a real Ollama container pulling a real model and a
live response. Run it manually when verifying a release:

1. `docker compose up -d`
2. Open `/models`, create a container for a small model (e.g. `llama3.2:1b`)
3. Wait for it to reach "Running"
4. Open `/chat`, send a message, confirm a streamed response arrives

This isn't automated because it's slow, needs real disk space and network
access to ollama.com/registry.ollama.ai, and isn't something that should
run unattended on every CI push.
