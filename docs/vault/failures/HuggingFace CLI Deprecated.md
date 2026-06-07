---
tags: [failure, huggingface, cli, upload, p3]
---

# HuggingFace CLI Deprecated

## Symptom
Tried to upload checkpoints to HuggingFace using:
```bash
huggingface-cli upload mitvho09/humanoid-g1-nav ...
```
Got:
```
zsh: command not found: huggingface-cli
```
or (in older sessions):
```
huggingface-cli: error: argument command: invalid choice: 'upload'
```

## Root Cause
Newer versions of `huggingface_hub` deprecated the `huggingface-cli` command-line tool. The Python API (`HfApi`) is the current standard.

Additionally, on Lightning Studios, `huggingface-cli` is not in the default PATH even when the Python package is installed.

## Fix: Use Python API Directly
```bash
/home/zeus/miniconda3/bin/python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='hf_...')

# Upload a single file
api.upload_file(
    path_or_fileobj='/path/to/model_499.pt',
    path_in_repo='checkpoints/p3_vision_nav/run_300_l4/model_499.pt',
    repo_id='mitvho09/humanoid-g1-nav',
    repo_type='dataset'
)

# Upload a whole folder (use this for checkpoints)
api.upload_folder(
    folder_path='/path/to/run_300_l4',
    repo_id='mitvho09/humanoid-g1-nav',
    repo_type='dataset',
    path_in_repo='checkpoints/p3_vision_nav/run_300_l4',
    commit_message='P3 final checkpoints'
)
print('Upload complete')
"
```

Run in background for large uploads:
```bash
nohup /home/zeus/miniconda3/bin/python3 -c "..." > /tmp/hf_upload.log 2>&1 &
echo "Upload PID: $!"
# Monitor:
tail -f /tmp/hf_upload.log
```

## If huggingface_hub Not Installed
```bash
/home/zeus/miniconda3/bin/pip install huggingface_hub -q
```

## Related
- [[Lightning Studio Environment]]
- [[PYTHONPATH & Python Interpreters]]
