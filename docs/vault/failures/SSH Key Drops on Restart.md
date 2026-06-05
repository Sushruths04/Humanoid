---
tags: [failure, ssh, lightning-ai, key, restart]
---

# SSH Key Drops on Restart

## Symptom
```
Permission denied (publickey,gssapi-keyex,gssapi-with-mic)
```

## Why
When a Lightning Studio goes to sleep or the machine changes, SSH key auth registration expires even though the key files still exist on disk.

## Fix (takes 30 seconds)

In PowerShell on your Windows machine:
```powershell
iwr "https://lightning.ai/setup/ssh-windows?t=7a558fd4-c340-44e0-accd-4b620bbbbf0e&s=01kt558jf0ra2chne251dtsg8k" -useb | iex
```

If that returns HTTP 500 (duplicate key), the key exists but the registration lapsed. Try the ssh-public endpoint directly:
```powershell
# Or just wait 30s and retry the SSH connection — sometimes it self-recovers
ssh s_01kt558jf0ra2chne251dtsg8k@ssh.lightning.ai
```

## Related
- [[Lightning Studio Environment]]
- [Lightning SSH Setup](../../docs/setup/LIGHTNING_SSH_SETUP.md)
