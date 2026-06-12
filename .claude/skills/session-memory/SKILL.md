---
name: session-memory
description: >
  Give Claude persistent memory across sessions so it stops forgetting prior work — automatic
  capture, AI compression, and context injection at the start of each new session. Use when the user
  says the agent "keeps forgetting", wants memory/continuity across sessions, asks to set up
  claude-mem or a "memory system", wants past sessions' context auto-loaded, or is choosing between
  memory layers. Covers installing/operating claude-mem (hook-driven plugin) and a lightweight
  manual fallback. Complements knowledge-base-reflector: this is automatic raw memory, that is
  curated lessons.
allowed-tools: Bash, Read, Write, Edit
---

# Session Memory

Claude Code sessions are stateless by default — each one starts blank. A session-memory layer fixes
that: it **captures** what happened, **compresses** it with AI into semantic summaries, and
**injects** the relevant pieces back into future sessions automatically.

> Derived from: "I Built The Best Claude Memory System (Beats Hermes)" — built around **claude-mem**
> (`github.com/thedotmack/claude-mem`).

## When to use this vs. the reflector skill

| | `session-memory` (this) | `knowledge-base-reflector` |
|--|--|--|
| What it stores | Everything the agent did, auto-captured + compressed | Hand-picked durable lessons/rules |
| How | Hooks fire automatically every session | You consciously reflect and write entries |
| Best for | Continuity ("what did we do last time?") | Standards, gotchas, preferences |

They are complementary — run both. claude-mem gives broad recall; the reflector keeps a clean,
trusted rulebook.

## Mode A — claude-mem (the plugin)

A Claude Code plugin that uses the Claude Agent SDK to compress tool-usage observations and a
background worker + vector store for retrieval.

```bash
npx claude-mem install     # or use the /plugin command in Claude Code
```

**How it works (lifecycle hooks):** `Setup → SessionStart → UserPromptSubmit → PreToolUse →
PostToolUse → Stop`.
- **SessionStart** ensures the worker is running and fetches historical memory into the context window.
- **PreToolUse/PostToolUse** capture tool-usage observations during the session.
- **Stop** compresses the session into semantic summaries for later retrieval.

**Storage / config (defaults):**
- SQLite DB: `~/.claude-mem/claude-mem.db`
- Chroma vector store: `~/.claude-mem/chroma/`
- Worker port: `37700 + (uid % 100)` — override with `CLAUDE_MEM_WORKER_PORT`
- Data dir: override with `CLAUDE_MEM_DATA_DIR`

**Operational notes & honest caveats:**
- Requires Node/`npx` and network access to install; the SessionStart hook spawns a worker — if it
  errors, sessions can be slow to start. Disable the plugin to isolate startup problems.
- The hooks run on *your* machine. In an **ephemeral cloud session** (Claude Code on the web),
  `~/.claude-mem` lives in a throwaway container and is wiped when it's reclaimed — memory won't
  persist unless `CLAUDE_MEM_DATA_DIR` points at committed/mounted storage. claude-mem shines in a
  persistent local install; treat it cautiously in disposable environments.
- It works across agents (Claude Code, Codex, Gemini, OpenCode, etc.), so memory is portable.

## Mode B — lightweight manual fallback (no plugin / locked-down env)

When you can't run the plugin, emulate the loop with a committed file:

1. **Inject at start:** `Read` `.claude/state/session-log.md` and apply relevant prior context.
2. **Capture at end:** append a compressed summary of the session — goal, key decisions, files
   touched, open threads, next steps. Keep it short; this is recall, not a transcript.
3. **Commit it** so it survives container churn and travels with the repo.

```markdown
## <YYYY-MM-DD> — <session goal>
- Did: <what changed / decisions>
- Files: <key paths>
- Open: <unfinished threads / next step>
```

This is poor-man's claude-mem: manual capture + start-of-session injection, but durable and zero-dep.
