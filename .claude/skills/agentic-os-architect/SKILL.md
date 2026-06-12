---
name: agentic-os-architect
description: >
  Scaffold, audit, and maintain an "agentic OS" inside a repo — an organized .claude/ system of
  skills, subagents, slash commands, shared context (CLAUDE.md), and file-based state — so Claude
  operates as one coherent, self-maintaining system rather than ad-hoc prompts. Use when the user
  wants to set up, restructure, or review their Claude Code setup, asks to "build an agentic OS",
  organize skills/agents, establish project standards and workflows, or wire individual skills into
  a larger system with shared context and a learning loop.
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
---

# Agentic OS Architect

An "agentic OS" is the 2026 control-stack pattern: **project rules + reusable skills + bounded
subagents + deterministic tools**, all sharing context and improving over time. The goal is a system
that is organized, predictable, and self-maintaining — not a pile of one-off prompts.

> Derived from: "Stop Using Claude Code Without an Agentic OS" (`brobertsaz/claude-os`) and
> "Creating Your Own Agentic OS is Easy" — both build the same layered structure.

## Recommended layout

```
CLAUDE.md                     # always-loaded: standards, commands, architecture, conventions
.claude/
  skills/<name>/SKILL.md      # reusable, auto-triggered workflows (model decides when to load)
  commands/<name>.md          # explicit slash commands the user invokes by name
  agents/<name>.md            # subagent definitions (own context window + scoped tools)
  knowledge/                  # persistent memory / learnings (see knowledge-base-reflector)
  state/*.json|*.md           # file-based state: tasks, specs, logs (no vector DB needed)
```

Memory is **file-based** (JSON + markdown). You do not need a vector DB, Redis, or Postgres to start.

## The five layers (and the principle behind each)

1. **Shared context** — `CLAUDE.md` auto-loads every session; sub-directory `CLAUDE.md` files give
   specialists their own rules. Keep it lean; it is always in the context window.
2. **Skills** — small markdown folders Claude pulls in on demand via a one-line description.
   Principle: *progressive disclosure* — load detail only when needed.
3. **Subagents** — bounded child contexts for parallel/isolated work. Principle: *context isolation*
   — give each a narrow job and only the tools it needs.
4. **Deterministic tools** — scripts/commands the agent calls instead of reasoning from scratch.
   Principle: *put determinism around the model* so results are reproducible.
5. **Learning loop** — sessions reflect and write lessons back (pairs with `knowledge-base-reflector`)
   so the OS improves with use.

## To scaffold (new repo)

1. Create the layout above. Seed `CLAUDE.md` with: build/test/lint commands, key directories,
   coding standards, and "how to work here" notes.
2. Add a `state/` convention (e.g. `tasks.md`, `specs/`) for multi-step work.
3. Register the starter skills the project needs; cross-link them in `CLAUDE.md`.
4. **Commit everything.** In ephemeral cloud sessions, only what's in git survives.

## To audit (existing repo)

- Is there a `CLAUDE.md`? Is it lean and accurate, or stale/bloated?
- Do skills have tight, trigger-rich descriptions (so they actually fire)?
- Are subagents scoped to minimal tools, or over-privileged?
- Is there a learning loop, or do corrections evaporate each session?
- Report findings as a short prioritized checklist; fix the high-value gaps.

## Principles to enforce

- Portable and committed over clever-but-local.
- One responsibility per skill/agent; compose them, don't merge them.
- Prefer deterministic scripts for anything done the same way every time.
- Lean shared context; deep detail lives in on-demand files.
