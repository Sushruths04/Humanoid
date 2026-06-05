---
tags: [failure, docker, ghcr, auth, token]
---

# GHCR Auth Denied

## Symptom
```
Error response from daemon: Head "https://ghcr.io/v2/sushruths04/humanoid-isaaclab/manifests/latest": denied: denied
```
or: `unauthorized: unauthorized`

## Root Cause
The stored GHCR token (`ghp_XchPcb...`) in `~/.docker/config.json` was revoked — it was pasted into the chat conversation and therefore had to be revoked immediately for security. On the next machine restart (with a fresh docker config), the anonymous pull also failed because the package was still private.

## Fix: Make the Package Public
```
github.com/users/Sushruths04/packages/container/humanoid-isaaclab/settings
→ Danger Zone → Change visibility → Public
→ type "humanoid-isaaclab" → click the red confirm button
```

Test it worked:
```bash
curl -s "https://ghcr.io/token?scope=repository:sushruths04/humanoid-isaaclab:pull" \
  -w "HTTP %{http_code}\n" -o /dev/null
# HTTP 200 = public ✅   HTTP 401 = still private ❌
```

**If you need it private**: create a new classic PAT at `github.com/settings/tokens` with `read:packages` scope only:
```bash
docker login ghcr.io -u Sushruths04 --password-stdin <<< "<new_PAT>"
```

## Prevention
- Never paste long-lived tokens (GitHub PATs, HF tokens) into chat.
- For shared dev images, make the package public — it's simpler and eliminates auth headaches across machine restarts.
- If you must keep it private, store the token in a persistent file (e.g. `~/.ghcr_token`) on the persistent `/teamspace/` filesystem, not in chat.

## Related
- [[GHCR Image & Auth]]
- [[Isaac Sim Docker Container]]
- [[SSH Key Drops on Restart]]
