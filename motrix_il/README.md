# motrix_il

`motrix_il` is the imitation learning subproject for MotrixLab. It uses `uv`
to manage a dedicated Python 3.12 environment and vendors the LeRobot source
tree under `motrix_il/src/lerobot-main`.

## Why this is a separate project

The current MotrixLab workspace targets Python 3.10, while the vendored
LeRobot version requires Python 3.12 or newer. Keeping `motrix_il` separate
avoids breaking the existing `motrix_envs` and `motrix_rl` workflows.

## Usage

Create the Python 3.12 environment:

```bash
uv sync --project motrix_il
```

Install the vendored LeRobot training stack into `motrix_il/.venv`:

```bash
./motrix_il/.venv/bin/python -m ensurepip --upgrade
./motrix_il/.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e './src/lerobot-main[training]'
./motrix_il/.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
```

Enable extra policy stacks when needed:

```bash
./motrix_il/.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e './src/lerobot-main[diffusion]'
./motrix_il/.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e './src/lerobot-main[pi]'
```

Run code inside the `motrix_il` environment:

```bash
./motrix_il/.venv/bin/python -c "import motrix_il; print(motrix_il.describe())"
./motrix_il/.venv/bin/lerobot-train --help
```

After the environment is installed, prefer maintaining it with
`./motrix_il/.venv/bin/python -m pip ...`. Avoid running
`uv sync --project motrix_il` again, because `uv.lock` may try to reconcile
the environment to a different dependency graph than the verified
`pip`-managed setup.
