---
name: model-router
description: >
  Pick the right Claude model and prompt it correctly for the task at hand. Use when the user asks
  which model to use, mentions Fable 5 / Opus / Sonnet / Haiku, is trading off cost vs quality vs
  speed, is getting poor results that look like a model mismatch, or is assigning models to agents /
  subagents / pipeline stages. Encodes per-model strengths and prompting tactics. For authoritative
  model IDs, pricing, context limits, and API parameters, defer to the claude-api skill.
allowed-tools: Read
---

# Model Router

Most "bad Claude output" is a routing problem, not a capability problem: the wrong model for the job,
or the right model prompted like an older one.

> Derived from: "You are using Claude Fable 5 wrong" — the recurring mistake is treating a newer
> flagship like a legacy model: over-scaffolding prompts, withholding tools, or aiming it at trivial
> work where a cheaper model wins.

## Current models (June 2026)

| Model | ID | Sweet spot |
|-------|----|-----------|
| **Fable 5** | `claude-fable-5` | Latest flagship — creative, multimodal, and agentic work; let it drive a loop |
| **Opus 4.8** | `claude-opus-4-8` | Hardest reasoning, large refactors, architecture, long-context (1M) |
| **Sonnet 4.6** | `claude-sonnet-4-6` | Balanced default for most coding/agentic tasks; strong cost/quality |
| **Haiku 4.5** | `claude-haiku-4-5-20251001` | Fast, cheap, high-volume: classification, extraction, simple edits |

> Treat the table as routing guidance. **Confirm exact IDs, pricing, and limits via the `claude-api`
> skill** before hard-coding them — do not quote prices from memory.

## How to route

- **Triage by difficulty, not by default.** Simple/narrow/high-volume → Haiku. Everyday coding and
  tool-use → Sonnet. Genuinely hard reasoning, big refactors, whole-system design, or very long
  context → Opus. Frontier creative/multimodal/agentic → Fable 5.
- **Cost/quality/speed is a triangle** — name which one the user is optimizing and pick accordingly.
- **In multi-agent setups:** put the expensive model on the orchestrator / hard-reasoning step and
  cheap models on the many narrow subagent tasks. Don't pay Opus rates to rename variables.

## How to prompt each well

- **Newer flagships (Fable 5, Opus 4.8):** give the goal and the tools, then get out of the way.
  Over-specifying every step and wrapping them in legacy scaffolding *reduces* quality. Let them plan
  and use an agentic loop. Provide context and constraints, not a rigid script.
- **Sonnet:** clear task + acceptance criteria; reliable workhorse for structured execution.
- **Haiku:** be explicit and bounded — exact output format, one job, minimal reasoning surface.

## Common mismatches to catch

- Using a flagship for bulk classification (wastes money) — route to Haiku.
- Using Haiku for multi-step architectural reasoning (wastes quality) — route to Opus.
- Caging a flagship with step-by-step micromanagement and no tools (wastes the model) — loosen the
  prompt, grant tools, set a goal.
