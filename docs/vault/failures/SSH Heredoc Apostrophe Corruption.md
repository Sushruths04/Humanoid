---
tags: [failure, ssh, heredoc, quoting, file-corruption]
---

# SSH Heredoc Apostrophe Corruption

## Symptom
After writing a Python file over SSH using `cat <<'EOF' ... EOF`, the file has a SyntaxError:
```
SyntaxError: unterminated triple-quoted string literal (detected at line 47)
```
or the file gets truncated mid-way through.

## Root Cause
When using `ssh host 'cat <<'"'"'EOF'"'"' ... EOF'` (single-quoted heredoc inside single-quoted SSH command), any apostrophe (`'`) in the file content **terminates the outer single-quote context**. Everything after it is interpreted as shell commands, not file content.

Example: the docstring `"""Entry k is the first timestep at which subgoal k's marker was reached"""` — the apostrophe in `k's` ends the heredoc prematurely.

## Fix: Transfer Code as Files (scp)

Write the file locally, then use `scp`:
```bash
# 1. Write the file locally (in Windows):
#    D:\Mini Thesis\NVIDIA\_xfer_myfile.py

# 2. scp it to the Studio:
scp "D:\Mini Thesis\NVIDIA\_xfer_myfile.py" \
  s_01kt558jf0ra2chne251dtsg8k@ssh.lightning.ai:/teamspace/studios/this_studio/Humanoid/programs/common/eval/myfile.py

# 3. Verify:
ssh ... 'python -m py_compile /path/to/myfile.py && echo OK'
```

scp handles binary transfer with no shell quoting interpretation. Any content — apostrophes, double quotes, backslashes — transfers perfectly.

## The Method Used This Session
All code files were written locally using the `Write` tool, then `scp`'d to the Studio. No SSH heredocs for any code. This became the standard practice after this failure.

## Lesson
> **scp for code, SSH for commands.** Never inline code through nested shell quoting.

If you ever need to write a small file over SSH (e.g., a one-liner config), use printf with hex escapes or base64 encoding — but scp is always cleaner for anything with real content.

## Related
- [[Lightning Studio Environment]]
- [[PYTHONPATH & Python Interpreters]]
