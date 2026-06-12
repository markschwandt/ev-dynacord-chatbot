---
name: codebase-knowledge-graph
description: >
  Build and query a knowledge graph of a codebase or document corpus to understand structure,
  dependencies, and the "why" behind the design — instead of fuzzy text/embedding search. Use when
  the user wants to map, explore, or onboard onto an unfamiliar repo, trace how components connect,
  assess the blast radius of a change, or asks to "build a knowledge graph", "graphify", "map the
  codebase", or understand the EV/Dynacord document corpus as a graph. Produces an explicit graph of
  entities and relationships you can traverse like a senior engineer builds a mental model.
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Codebase Knowledge Graph

Embedding search retrieves by similarity; a knowledge graph retrieves by **traversal** — it captures
how entities relate (calls, imports, defines, depends-on, documents) and *why*. That is closer to
how an engineer navigates an unfamiliar system.

> Derived from: "The INSANE Claude Code Knowledge Graph Stack", which centers on **Graphify**
> (`github.com/safishamsi/graphify`).

## Mode A — Graphify (preferred when the network allows)

Graphify combines Tree-sitter static analysis with LLM semantic extraction.

```bash
# Install (PyPI package is "graphifyy" — double y; CLI is "graphify")
pipx install graphifyy   # or: pip install graphifyy

graphify <path>                     # full pipeline on a directory
graphify https://github.com/o/repo  # clone then graph a remote repo
```

Outputs land in `graphify-out/`:
- `graph.html` — interactive graph (clickable nodes, filters, search)
- `GRAPH_REPORT.md` — highlights, key concepts, surprising connections, suggested questions
- `graph.json` — the full graph; **query this without re-reading source files**

Every edge is tagged **EXTRACTED**, **INFERRED**, or **AMBIGUOUS** — trust accordingly. Auto-sync
mode + a post-commit git hook keep the graph fresh as code changes.

To answer a question after a build: `Read` `GRAPH_REPORT.md` first, then `Grep`/parse `graph.json`
for the relevant nodes and traverse their edges — don't re-scan the whole tree.

> ⚠️ Graphify's semantic extraction calls an LLM and may need network/API access. In a locked-down
> environment (egress allowlist) the LLM step can fail — fall back to Mode B, or run Graphify where
> network is available and commit `graphify-out/` so the graph travels with the repo.

## Mode B — Lightweight graph (offline / no external deps)

When Graphify cannot run, build a focused graph yourself:

1. Enumerate entities with `Glob`/`Grep` (modules, classes, functions, data files, doc chunks).
2. Extract edges from the source: imports, function calls, file reads/writes, schema references.
3. Write `knowledge-graph.json` as `{ "nodes": [...], "edges": [{from, to, type, confidence}] }`.
4. Answer questions by traversing that JSON; tag inferred edges so the user knows what was guessed.

## Applying to this repo (EV/Dynacord corpus)

The corpus (`data/chunks/all_chunks.json`, 63k chunks across 3,080 PDFs) is a natural graph: nodes =
products / documents / chunks; edges = "same product", "same brand", "references model X",
"belongs-to category". A graph over it enables structural queries ("all firmware docs for Dynacord
mixers") that TF-IDF similarity alone misses. Build the graph from the chunk metadata, not the raw
text, to keep it cheap.
