---
tags: [failure, ssh, heredoc, python, file-write, p3]
---

# Python File Corruption Over SSH — Use Python Write

## Symptom
Wrote `custom_play.py` to the remote machine using SSH heredoc:
```bash
ssh user@host "cat > /path/custom_play.py << 'EOF'
import os, sys
...
EOF"
```
Running it gave:
```
SyntaxError: invalid syntax
      ^^^^^
There was an error running python
```
The file on disk had mangled string literals and broken escape sequences.

## What Was Tried
First attempt: SSH heredoc with `<<'EOF'` (single-quoted to prevent variable expansion). Still corrupted — escape sequences like `\"` and `\n` inside Python string literals were processed by the shell.

Second attempt: using `cat << 'EOF'` inside a larger shell command. Same corruption due to nested quoting.

## Root Cause
When composing `ssh 'command with heredoc'`, the outer single quotes and inner heredoc quoting interact unpredictably across different shells and SSH implementations. Escape sequences, double quotes inside strings, and backslashes are all processed at multiple shell-expansion layers before reaching the file.

This is an extension of [[SSH Heredoc Apostrophe Corruption]] but affects **any special character** in Python code (not just apostrophes): `\"`, `\n`, f-string braces `{}`, etc.

## Fix: Write via Python -c
Use `python3 -c` with a string assignment to write the file — Python handles all escaping:

```bash
ssh user@host "/home/zeus/miniconda3/bin/python3 -c \"
content = '''import os, sys
try:
    import my_humanoid_project.tasks
except ImportError as e:
    print(f\\\"Error: {e}\\\")
    import sys; sys.exit(1)
...
'''
with open('/path/custom_play.py', 'w') as f:
    f.write(content)
print('written')
\""
```

Then verify:
```bash
ssh user@host "/home/zeus/miniconda3/bin/python3 -m py_compile /path/custom_play.py && echo 'syntax OK'"
```

## Best Fix: Write Locally, scp
The cleanest approach (as established by [[SSH Heredoc Apostrophe Corruption]]):
1. Write the file locally using the `Write` tool
2. `scp` it to the remote machine

```bash
scp -i ~/.ssh/lightning_rsa \
  "D:\Mini Thesis\NVIDIA\my-humanoid-project\custom_play.py" \
  s_XXXX@ssh.lightning.ai:/teamspace/studios/this_studio/Humanoid/my-humanoid-project/custom_play.py
```

## Rule
> **Never write Python code via SSH heredoc or shell echo.** Always either `scp` from local or use `python3 -c "with open(...) as f: f.write(...)"`.

## Related
- [[SSH Heredoc Apostrophe Corruption]]
- [[play.py Fails - Custom Task Not Registered]]
- [[Lightning Studio Environment]]
