# Media folder — drop your videos/images here

The site shows placeholders until you add files here. To make a video appear, just save it
with the exact name below (no code changes needed — the HTML already points at it):

| File name | Shows up in | Where to get it |
|---|---|---|
| `robust_rollout.mp4` | Chapter 6 (Results) | export the G1 robust-locomotion rollout (Isaac Lab `record_video` / your `thesis/results`) |
| `wakeboard_rollout.mp4` | Chapter 8 (Wakeboarding) | produced by `wakeboarding-experiment/scripts/40_record_video.sh` after GPU runs |
| `poster.jpg` | Chapter 6 video poster | any still frame (optional) |

Tips:
- Keep clips short (5–15 s) and compressed (H.264 .mp4) so the folder stays shareable.
- Images work too — add `<img src="../media/your_image.png">` wherever you like.
- Everything is **relative paths**, so the whole `Humanoid_Docs_Site/` folder is self-contained:
  zip it, send it, open `index.html` on any laptop — videos included, no internet needed.
