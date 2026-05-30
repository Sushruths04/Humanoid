# STEP 01 GR00T install deferred

_2026-05-30T18:48:28Z_

## Deferred to explicit CPU install

This step is CPU-compatible but downloads and installs a large upstream repo.
To run it, edit `thesis/config.env`:

```bash
RUN_CPU_INSTALLS=1
```

Then execute:

```bash
bash thesis/run_thesis.sh --all --from 01_gr00t_install
```

# STEP 01 GR00T install

_2026-05-30T18:58:36Z_

## Commands

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
/home/zeus/miniconda3/envs/thesis310/bin/python -m pip install -e .
python3 -c "import gr00t; print('ok')"
```

## Verification

```text
$ /home/zeus/miniconda3/envs/thesis310/bin/python -c import\ gr00t\;\ print\(\'ok\'\)
ok
```

