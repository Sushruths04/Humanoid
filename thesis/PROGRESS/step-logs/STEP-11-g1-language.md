# STEP 11 G1 language-conditioning code

_2026-05-30T18:23:06Z_

## Authored package

- `my-humanoid-project/my_humanoid_project/language_commands.py`
- `my-humanoid-project/my_humanoid_project/tasks/g1_language_pickplace_cfg.py`
- `my-humanoid-project/my_humanoid_project/tasks/__init__.py`

## Verification

```text
$ python3 -c from\ my_humanoid_project.language_commands\ import\ COMMANDS\,\ embedding_for_text\;\ print\(len\(COMMANDS\)\)\;\ print\(len\(embedding_for_text\(\'pick\ up\ the\ red\ cube\'\)\)\)
4
16
```

- [x] CPU import works without Isaac Sim
- [x] Deterministic language embeddings are available
- [ ] GPU validation: register env inside Isaac Lab and run random-policy step

