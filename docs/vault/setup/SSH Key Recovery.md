---
tags: [setup, ssh, lightning-ai, recovery]
---

# SSH Key Recovery

## Symptom

```
Permission denied (publickey)
```

## Why It Happens

When a Lightning Studio goes to sleep or the machine restarts, the SSH key authentication is dropped. The key itself still exists on the Studio but the auth registration expires.

---

## Fix — Re-download the SSH Key

Run this in PowerShell on your local Windows machine (the Lightning setup script):

```powershell
# Token and Studio ID for Sushruth's setup:
# Token:     t=7a558fd4-c340-44e0-accd-4b620bbbbf0e
# Studio ID: id=60464254-02e0-4267-9809-3fe4d0ab6b92

iwr "https://lightning.ai/setup/ssh-windows?t=7a558fd4-c340-44e0-accd-4b620bbbbf0e&s=01kt558jf0ra2chne251dtsg8k" -useb | iex
```

> **Note:** `ssh-gen` endpoint returns HTTP 500 (duplicate) if the key already exists — that's fine, not an error. The `ssh-public` endpoint still works and re-registers the key.

After running, reconnect:
```bash
ssh s_01kt558jf0ra2chne251dtsg8k@ssh.lightning.ai
```

---

## Full SSH Setup (first time)

Full guide: [Lightning SSH Setup](../../docs/setup/LIGHTNING_SSH_SETUP.md)

---

## Related

- [[Lightning Studio Environment]]
- [Lightning Backup Workflow](../../LIGHTNING_BACKUP_WORKFLOW.md)
- [Recovery Guide](../../RECOVERY_GUIDE.md)
