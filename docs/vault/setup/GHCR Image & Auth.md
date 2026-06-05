---
tags: [setup, docker, ghcr, auth, github]
---

# GHCR Image & Auth

## Current Status

The image `ghcr.io/sushruths04/humanoid-isaaclab:latest` is **PUBLIC** (since 2026-06-05). You can pull it anonymously with no login.

**Quick test (from anywhere):**
```bash
curl -s "https://ghcr.io/token?scope=repository:sushruths04/humanoid-isaaclab:pull" \
  -w "HTTP %{http_code}\n" -o /dev/null
# HTTP 200 = public and pullable
# HTTP 401 = private, need auth
```

---

## Pulling (anonymous, no token needed)

```bash
docker logout ghcr.io   # ensure no stale credentials interfere
docker pull ghcr.io/sushruths04/humanoid-isaaclab:latest
docker tag  ghcr.io/sushruths04/humanoid-isaaclab:latest isaac-lab-base
```

---

## If It's Private (how to auth)

Option A — login with a PAT:
```bash
# Create a classic PAT at github.com/settings/tokens with read:packages scope
docker login ghcr.io -u Sushruths04 --password-stdin <<< "<your_PAT>"
docker pull ghcr.io/sushruths04/humanoid-isaaclab:latest
```

Option B — make the package public (recommended, done once):
```
github.com/users/Sushruths04/packages/container/humanoid-isaaclab/settings
→ Danger Zone → Change visibility → Public
→ Type "humanoid-isaaclab" to confirm → click the red button
```
The change is instant. Test with the `curl` command above.

> **Important:** always rotate any PAT pasted into chat immediately. The old `ghp_XchPcb...` token was exposed and is revoked.

---

## Why We Made It Public

Docker storage is ephemeral (wiped on machine restart). Every fresh GPU session needs to re-pull 17.6 GB. With a private image you need valid credentials every time — which breaks if a token is revoked. Making it public eliminates this friction entirely for a dev/learning image.

---

## Related

- [[Isaac Sim Docker Container]]
- [[GHCR Auth Denied]] (failure note)
- [[Lightning Studio Environment]]
