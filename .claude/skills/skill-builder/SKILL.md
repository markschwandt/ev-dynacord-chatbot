---
name: skill-builder
description: >
  Author a new, well-structured Claude skill (SKILL.md + optional supporting files) that triggers
  reliably and follows progressive-disclosure best practices. Use when the user wants to create,
  scaffold, improve, or debug a skill, asks to "make a skill for X", wants a skill to auto-trigger
  more reliably, is building out a fleet of skills, or is reviewing existing skills for quality.
  This is the meta-skill that lets the fleet keep growing.
allowed-tools: Read, Write, Edit, Glob, Grep
---

# Skill Builder

A skill is a folder Claude loads **on demand** when its `description` matches the situation. Two
things make a skill good: a description that triggers at the right time, and a body that is lean,
actionable, and pushes detail into on-demand files.

## Anatomy

```
.claude/skills/<name>/
  SKILL.md          # required: frontmatter + instructions
  reference.md      # optional: deep detail, loaded only when SKILL.md points to it
  scripts/*.py|sh   # optional: deterministic helpers the skill calls
```

### Frontmatter

```yaml
---
name: my-skill            # lowercase, hyphens, MUST match the directory name
description: >            # the single most important field — see below
  <what it does> + <exactly when to use it> + <trigger words the user might say>
allowed-tools: Read, Edit # optional: restrict to the tools the skill needs
---
```

## Writing the description (this is what makes it fire)

The model reads only the `description` to decide whether to load the skill. Make it earn the trigger:

- **Third person, about the skill** ("Use when…", not "I will…").
- Pack in **WHAT it does**, **WHEN to use it**, and **concrete trigger phrases** the user might type.
- Cover synonyms and the real situations — vague descriptions never fire; specific ones do.
- Keep it under ~1024 characters. One dense paragraph beats a sentence.

> Good: "Build and query a knowledge graph of a codebase… Use when the user wants to map a repo,
> trace dependencies, or asks to 'graphify' / 'map the codebase'."
> Weak: "Helps with codebases." (no triggers, never loads)

## Writing the body (progressive disclosure)

- Keep `SKILL.md` focused — it loads fully into context when triggered, so don't bloat it.
- Lead with the workflow: numbered steps the model can execute.
- Push long references, schemas, or examples into sibling files and **link** to them ("see
  `reference.md`") so they load only when needed.
- Prefer deterministic scripts for anything done the same way every time; have the skill call them.
- State preconditions and failure fallbacks explicitly.

## Build process

1. Clarify the **one job** this skill does. If it has two jobs, make two skills.
2. Draft the description with triggers; sanity-check: "would this fire on the phrasings the user
   actually uses?"
3. Write the workflow body; extract detail into reference files.
4. Name the dir == `name`. Place under `.claude/skills/` (project) — and commit it.
5. **Test the trigger:** describe 2–3 realistic user phrasings and confirm the description matches.
   Test a near-miss that should *not* trigger.

## Quality checklist

- [ ] `name` matches directory, lowercase-hyphenated
- [ ] Description has WHAT + WHEN + trigger phrases, third person, < ~1024 chars
- [ ] Single responsibility
- [ ] Body is steps-first; heavy detail offloaded to reference files
- [ ] `allowed-tools` scoped to what's needed (omit to allow all)
- [ ] Committed to git
