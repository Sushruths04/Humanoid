# Humanoid RL — Interactive Docs (HTML)

A self-contained, offline, course-style website explaining the whole Humanoid RL project —
for understanding it yourself and for showing colleagues / interviewers.

## How to open
**Just double-click `index.html`** in any browser. No server, no internet, no install.

## How to share
Zip the entire `Humanoid_Docs_Site/` folder and send it. It works on any laptop because
everything uses **relative paths** (HTML + one CSS + one JS + your media). Nothing is fetched
from the internet.

## What's inside
```
index.html              Home / overview / status
chapters/01..12.html    12 chapters (big picture → glossary)
assets/style.css        the futuristic dark theme
assets/site.js          shared sidebar nav + interactions
media/                  drop videos/images here (see media/README.md)
```

## Chapters
1. The Big Picture · 2. The Tech Stack · 3. Sit → Stand · 4. Reward Design ·
5. Simulation Setup · 6. Evaluation & Results · 7. Language & Vision ·
8. Wakeboarding RL · 9. Why These Choices · 10. Compute & Workflow ·
11. Roadmap · 12. Glossary

## Notes
- Written to be understandable by a non-expert, but every chapter has a **🎤 Interview answer**
  box with the precise, technical version.
- Grounded in the real codebase and results (e.g. robust locomotion 981/1000, GR00T MSE 25.9).
- To add a demo video: drop `robust_rollout.mp4` / `wakeboard_rollout.mp4` into `media/`.
- To edit the look: everything is in `assets/style.css` (CSS variables at the top).
- To add a chapter: copy a file in `chapters/`, then add one line to the `CHAPTERS` array in `assets/site.js`.
