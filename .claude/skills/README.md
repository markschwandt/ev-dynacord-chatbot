# Skills Fleet

A focused set of Claude Code skills derived from a YouTube playlist on building Claude into a
self-improving, well-organized agentic system. Each skill is a reusable workflow, not a video
summary. Skills auto-trigger when their `description` matches the situation; you can also invoke one
explicitly by name.

## The fleet

| Skill | What it does | Source video |
|-------|--------------|--------------|
| [`knowledge-base-reflector`](knowledge-base-reflector/SKILL.md) | Reflect→abstract→generalize→record lessons into persistent memory so sessions compound | Build a Self-Improving Claude Knowledge Base with ONE Prompt |
| [`codebase-knowledge-graph`](codebase-knowledge-graph/SKILL.md) | Build/query a knowledge graph of a repo or document corpus (Graphify + offline fallback) | The INSANE Claude Code Knowledge Graph Stack |
| [`agentic-os-architect`](agentic-os-architect/SKILL.md) | Scaffold/audit an "agentic OS": skills + subagents + commands + shared context + state | Stop Using Claude Code Without an Agentic OS · Creating Your Own Agentic OS is Easy |
| [`creative-asset-pipeline`](creative-asset-pipeline/SKILL.md) | Brief→strategy→copy→generated assets, tracked & reproducible (Adobe/Higgsfield tools) | Higgsfield Just Turned Claude Into a Creative Agency |
| [`model-router`](model-router/SKILL.md) | Pick the right Claude model and prompt it correctly (Fable 5 / Opus / Sonnet / Haiku) | You are using Claude Fable 5 wrong |
| [`session-memory`](session-memory/SKILL.md) | Persistent cross-session memory: auto-capture, AI compression, context injection (claude-mem + manual fallback) | I Built The Best Claude Memory System (Beats Hermes) |
| [`motion-website-builder`](motion-website-builder/SKILL.md) | Build a scroll-driven cinematic "motion website": AI motion clips wired into a Claude-built animated site | Claude Fable 5 + Higgsfield MCP Built This Motion Website |
| [`skill-builder`](skill-builder/SKILL.md) | Meta-skill: author new skills that trigger reliably (lets the fleet keep growing) | (capstone — serves the goal of building a fleet) |

> `session-memory` and `knowledge-base-reflector` are complementary: the former is *automatic* raw
> session recall (claude-mem), the latter is *curated* durable lessons. Run both.

## Source playlist

`https://youtube.com/playlist?list=PLBKfaIiXl7okpCBm95n6VD394AmfQoMjD`

| ID | Title |
|----|-------|
| `K2BpNt3UBOQ` | Build a Self-Improving Claude Knowledge Base with ONE Prompt |
| `mWLDn49_8HA` | The INSANE Claude Code Knowledge Graph Stack |
| `Bgxsx8slDEA` | Stop Using Claude Code Without an Agentic OS |
| `xn6Z5PYyAIE` | Higgsfield Just Turned Claude Into a Creative Agency |
| `vjdHAWvVCP4` | You are using Claude Fable 5 wrong |
| `w0S-khYCaB4` | Creating Your Own Agentic OS is Easy (Insanely Powerful) |
| `N5JeyaqIa7c` | Claude Fable 5 + Higgsfield MCP Built This Motion Website *(identified from a user-supplied screenshot; exact ID→title match inferred, not verified against YouTube)* |

### Added separately (not in the original playlist)

| ID | Title | Skill |
|----|-------|-------|
| `H9BUkgDf5Y4` | I Built The Best Claude Memory System (Beats Hermes) | `session-memory` |

## Provenance & honesty note

This environment cannot reach YouTube (network egress is restricted to an allowlist), so these skills
were **not** built from verbatim transcripts. They were reconstructed from each video's title plus
web research into the tools and techniques each video covers (Graphify, claude-os/agent-os, the
self-improving-KB "reflect and add to CLAUDE.md" pattern, Higgsfield, and Claude model selection).

Where a skill states a concrete mechanic, it is grounded in the underlying tool's documentation. The
*emphasis* and framing are inferred from the titles. If a video makes a specific claim that differs
from what's here, send the transcript or a correction and the relevant skill will be updated.

One video (`N5JeyaqIa7c`) could not be identified — its skill is not yet built. Provide its title to
complete the fleet.
