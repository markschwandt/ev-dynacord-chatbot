# Learnings

Curated, durable lessons for this repo (the `knowledge-base-reflector` pattern). Read at the start
of work; keep entries generalized and de-duplicated. Confidence: EXTRACTED (told/observed directly) ·
INFERRED (deduced) · TENTATIVE (guess, verify later).

---

### Resolve YouTube video titles with WebSearch, not WebFetch/curl
- **Lesson:** This environment's network is an egress allowlist — only `github.com`, PyPI, and the
  Anthropic API respond; YouTube, Google, and proxy mirrors return `403`. `WebFetch`/`curl`/`yt-dlp`
  cannot reach them. `WebSearch` runs server-side and DOES work — use it to resolve a video ID to its
  title and to research the tools a video covers.
- **Because:** Building skills from a YouTube playlist that was otherwise unreachable.
- **Confidence:** EXTRACTED · **Added:** 2026-06-12

### Skills live in `.claude/skills/<name>/SKILL.md` and must be committed
- **Lesson:** A skill is a folder with a `SKILL.md` (YAML frontmatter `name` + `description`, then
  instructions). `name` must match the directory. The container is ephemeral, so anything not
  committed + pushed is lost when it's reclaimed.
- **Confidence:** EXTRACTED · **Added:** 2026-06-12

### User prefers work landed on `main` via PR + merge
- **Lesson:** Default to developing on the feature branch; when the user says "move to main", open a
  PR and merge it (leaves a clean record) rather than force-pushing. The user is non-expert on git
  terms — explain plainly.
- **Confidence:** INFERRED · **Added:** 2026-06-12

### This repo is an EV/Dynacord document search/chatbot
- **Lesson:** 63,167 text chunks from 3,080 Electro-Voice/Dynacord PDFs; TF-IDF vector store
  (`data/vectorstore/`), a Streamlit chatbot (`app/chatbot.py`), and a self-contained HTML search app.
  Memory/knowledge-graph/creative skills are especially relevant to this corpus.
- **Confidence:** EXTRACTED · **Added:** 2026-06-12
