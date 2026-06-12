---
name: creative-asset-pipeline
description: >
  Run an end-to-end creative/marketing asset pipeline: turn a brief into a strategy, write the copy,
  generate image/video assets, and track every asset in a sheet/log with reproducible prompts and
  job IDs. Use when the user wants marketing materials, ad creatives, product images, social or UGC
  content, brand assets, a campaign, or a "creative agency" workflow. Wires to whatever generation
  tools are available (Adobe Express/Firefly MCP image & video tools, Higgsfield CLI/MCP). Treats
  creative output as a reproducible production system, not one-off prompting.
allowed-tools: Read, Write, Edit, Bash, Glob
---

# Creative Asset Pipeline

Turn a single brief into a batch of on-brand assets through a repeatable pipeline with a tracking
layer — a "creative agency in a box". The discipline is what makes it production-grade: every
generation is logged with its prompt and job ID so it can be reproduced, audited, and improved.

> Derived from: "Higgsfield Just Turned Claude Into a Creative Agency". Higgsfield exposes 30+
> image/video models (Soul, Seedance, Kling, Veo, Flux, …). The CLI is the right default for batch
> jobs (the MCP loads all tools at once and burns context); reach for the MCP for interactive one-offs.

## Pipeline

1. **Brief → strategy.** Restate the brief (audience, goal, channel, brand voice, constraints).
   Propose a concrete creative strategy and the asset list before generating anything.
2. **Copy/script.** Write the headline/copy or shot script for each asset. Get sign-off if the user
   is present; otherwise proceed and flag assumptions.
3. **Generate.** Produce each asset with an available tool (see below). One prompt per row.
4. **Track.** Append every asset to `creative/assets.csv` (schema below) with status, result URL/path,
   prompt, model, and job ID.
5. **Review loop.** Mark approved/rejected; regenerate rejects with a noted prompt change. The log is
   the memory that lets the pipeline improve.

## Tracking schema (`creative/assets.csv`)

```
id,campaign,asset_type,brief,prompt,model,status,result_path_or_url,job_id,date
```

`status` ∈ `pending | generating | done | approved | rejected`. Never overwrite a row — append a new
version so the history is intact.

## Available generation tools

Check what is connected this session before assuming:
- **Adobe (Firefly/Express) MCP** — image editing/effects, background ops, video quick-cut/resize,
  document/template rendering, stock search. Good for editing, compositing, and brand templates.
  Call the Adobe init/mandatory-init step first per its tool requirements.
- **Higgsfield CLI/MCP** — text→image and text→video generation at scale; best for batch ad/UGC
  pipelines on a schedule. Prefer the CLI for batch.
- Fallback: if no generation MCP is connected, produce the **strategy, copy, shot lists, and prompts**
  as deliverables and tell the user exactly which tool to run them through.

## Brand consistency

- Pin a brand block (palette, logo usage, tone, do/don'ts) in `creative/brand.md` and feed it into
  every prompt. For this repo, that means Electro-Voice / Dynacord identity and the existing
  "Marketing Materials" category — keep product naming accurate.
- Reuse a fixed character/style seed across a campaign for visual coherence.

## Notes

- Log prompts verbatim — reproducibility is the whole point.
- Keep generated binaries out of git unless small/intended; commit the CSV log and prompts.
