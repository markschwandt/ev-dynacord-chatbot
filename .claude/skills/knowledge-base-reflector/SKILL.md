---
name: knowledge-base-reflector
description: >
  Capture durable lessons from the current session into a persistent knowledge base so future
  sessions start smarter. Use at the end of a task or session, after a correction or mistake, when
  the user says "remember this", "add to your knowledge", "reflect", "capture learnings", or "update
  CLAUDE.md", and whenever you discover a non-obvious fact, gotcha, convention, or preference worth
  keeping. Also use to read the knowledge base back at the START of work so accumulated lessons are
  applied. Turns one-off corrections into permanent, compounding memory.
allowed-tools: Read, Edit, Write, Glob, Grep
---

# Knowledge Base Reflector

A self-improving memory loop. The best AI workflows are not linear — they are loops where each
session's output improves the next session's input. Every correction you capture makes the system
permanently smarter.

> Derived from: "Build a Self-Improving Claude Knowledge Base with ONE Prompt" — the core trick is
> the phrase **Reflect → Abstract → Generalize → Record**.

## Where memory lives

| File | Purpose | Read when |
|------|---------|-----------|
| `CLAUDE.md` (repo root) | Stable project rules, conventions, commands, architecture facts | Auto-loaded every session |
| `.claude/knowledge/learnings.md` | Accumulating log of lessons, gotchas, and corrections | At the start of a task |
| `.claude/knowledge/<topic>.md` | Deep notes on one subsystem (e.g. `vectorstore.md`) | When that subsystem is touched |

Create these if missing. Keep `CLAUDE.md` lean (it is always in context) — promote a learning into
it only once it is proven and broadly relevant; otherwise it goes in `learnings.md`.

## Start of session: read back

Before non-trivial work, `Read` `CLAUDE.md` and `.claude/knowledge/learnings.md` (if present) and
apply what is relevant. Skip silently if they do not exist.

## The reflection loop (end of task / after a correction)

1. **Reflect** — What in this session was non-obvious? Where was I corrected, surprised, or wrong?
2. **Abstract** — Strip the one-off specifics. What is the underlying rule?
3. **Generalize** — Will this apply to future, different tasks? If no, do not record it.
4. **Record** — Append one entry. Update or delete a prior entry if this supersedes it (don't let
   the file accumulate contradictions).

## Entry format

```markdown
### <short imperative title>
- **Lesson:** <the generalized rule>
- **Because:** <the concrete trigger — what happened>
- **Confidence:** EXTRACTED (user told me) | INFERRED (I deduced it) | TENTATIVE (guess, verify later)
- **Added:** <YYYY-MM-DD>
```

Confidence tags matter: they let a future session trust EXTRACTED facts and re-verify TENTATIVE ones.

## Record vs. skip

- **Record:** stable preferences ("always run `ruff` before committing"), non-obvious gotchas
  ("the TF-IDF matrix is pickled, load with the matching scikit-learn version"), architectural
  decisions and their rationale, repeated corrections.
- **Skip:** anything obvious from the code, one-off task details, secrets, transient state.

## Hygiene

- Keep `learnings.md` curated, not append-only forever — merge duplicates, prune the stale.
- Never store secrets/credentials.
- When a TENTATIVE lesson is confirmed in a later session, promote it to EXTRACTED.
- Commit knowledge files to git — in an ephemeral cloud session, uncommitted memory is lost.
