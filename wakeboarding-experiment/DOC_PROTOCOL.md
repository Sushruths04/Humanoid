# Documentation Protocol (READ BEFORE CODING)

This project is **self-documenting**. The Obsidian vault in `vault/` must always reflect reality. These rules are binding for any human or AI working here.

## The three rules
1. **New/changed script → update `vault/02_Implementation/Scripts.md`** in the same change: what it does, key args/env-vars, inputs, outputs, how to run. One row per script.
2. **A run produces results → update `vault/03_Results/Results_Live.md`** (the comparison tables) **and** the status badge in `vault/00_INDEX.md`. Never let a checkpoint exist without a results row.
3. **Every training run → one dated entry in `vault/04_Log/Experiment_Log.md`**: date, stage, config file, git commit, `v_pull`, #envs, #iters, checkpoint path, headline metrics, and a one-line observation.

## Result provenance rule
A number only goes in `Results_Live.md` if it has a **source artifact on disk** (a `results/*.json` or a `logs/.../train.log`). Cite the file. (Same honesty rule the main Humanoid vault uses.)

## Checkpoint rule
Every checkpoint in the PLAN §10.1 ladder, when reached, is: (a) saved, (b) pushed to Hugging Face, (c) given a Results row + Log entry. Checkpoints are gitignored locally (large); only their metrics/paths live in the vault.

## Definition of "documented"
A feature/run is **not done** until its Scripts/Results/Log entries exist. Treat docs as part of the acceptance criteria, not an afterthought.
