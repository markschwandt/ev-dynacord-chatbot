---
name: motion-website-builder
description: >
  Build a scroll-driven, cinematic "motion website" — AI-generated motion/video clips wired into a
  Claude-built animated HTML/CSS/JS site end to end. Use when the user wants an animated or cinematic
  landing page, a scroll-animated site, video-background hero, motion-design web page, or says "build
  a motion website", "animated website", or "Higgsfield website". Takes a brand kit + business
  details + source imagery, generates motion clips (Higgsfield MCP/CLI or other video tools), and
  assembles a deployable scroll-animated site against a fixed design system.
allowed-tools: Read, Write, Edit, Bash, Glob
---

# Motion Website Builder

Turn a brand + a few images into a polished, scroll-animated cinematic website: AI generates the
motion footage, Claude writes the site that choreographs it. The whole pipeline — raw photo →
motion clip → deployed HTML — can run in well under an hour.

> Derived from: "Claude Fable 5 + Higgsfield MCP Built This Motion Website". The pattern mirrors
> Higgsfield's downloadable Claude skill that builds scroll-driven sites automatically. Pair with
> `model-router` (use a flagship like Fable 5 and let it drive the build loop) and
> `creative-asset-pipeline` (which can supply/track the assets).

## Inputs to collect first

- **Brand kit:** palette, logo, fonts, voice/tone, do/don'ts → pin in `site/brand.md`.
- **Business details:** what it sells, audience, sections, the one CTA.
- **Source imagery:** hero photo(s) / product shots to animate (or generate them).

Do not start building until these are settled — the design system depends on them.

## Step 1 — Generate motion assets

Use whatever video/image generation is connected this session (check first):
- **Higgsfield MCP/CLI (ideal):** 30+ models (Cinema Studio, Soul, Kling, Veo, Seedream…). Classic
  move: supply a **start frame + end frame** and prompt a cinematic pan/zoom; use motion brushes for
  directed action and time-remapping for slow-mo. Output up to 4K, ≤15s, any aspect ratio
  (16:9 / 9:16 / 1:1 / 4:5). CLI is best for batch; MCP for interactive.
- **Adobe (Firefly/Express) MCP:** image work + video quick-cut/resize/effects when Higgsfield isn't
  connected.
- **Fallback (no generation tool):** deliver the full site code with documented placeholders and the
  exact generation prompts + frame specs, so the user can drop clips in later.

Keep clips short and loopable for backgrounds. Log every prompt + job id (see `creative-asset-pipeline`).

## Step 2 — Build the site against a fixed design system

Define and reuse a design system so every build is consistent — put it in `site/design-system.md`:
type scale, spacing, color tokens, motion timing/easing, section templates.

Then assemble the site:
- Semantic HTML + CSS, mobile-first, accessible (respect `prefers-reduced-motion`).
- **Scroll choreography:** IntersectionObserver or a scroll library (GSAP ScrollTrigger / Framer
  Motion / Lenis for smooth scroll). Reveal-on-scroll, parallax, pinned sections, scrub-linked video.
- **Video backgrounds:** muted, autoplay, `playsinline`, `loop`; poster image + lazy-load; provide a
  static fallback for reduced-motion and slow connections.
- Performance: compress/encode clips (`ffmpeg` to web-friendly MP4/WebM), preload the hero, lazy-load
  the rest. Motion should never block first paint.

## Step 3 — Preview & deploy

- Serve locally to verify: `python3 -m http.server` (or the project's dev server) and check scroll
  behavior + reduced-motion.
- Deploy via a connected host if available (e.g. Netlify MCP) or output a self-contained folder the
  user can drop on any static host.

## Reusable skill-file mindset

Capture the design system + step order so re-runs are turnkey: same trigger, same inputs, same build
process, same look. That repeatability is what turns this from a one-off into a product.

## Caveats

- Generation tools and API access may not be connected in every session — confirm before promising
  rendered clips; otherwise ship code + prompts.
- Keep large video binaries out of git unless intended; commit the code, design system, and prompts.
