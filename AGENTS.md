# AGENTS.md

Guidance for any agent (Claude, GPT-4/5, Gemini, etc.) that contributes code or analysis within this repository.

## Purpose
- Keep responses consistent: short, actionable answers with references to files/lines, just like humans expect in PR reviews.
- Ship improvements quickly but safely: always favor minimal, well-tested changes over speculative refactors.
- Capture context so other agents (and humans) can resume work without re-reading the entire repo.

## Repo Overview
- `nanochat/`: Core model code (transformer, optimizers, tokenizer hooks, inference engine).
- `scripts/`: Entry points for training/eval/chat; most commands are thin wrappers around `nanochat.*`.
- `rustbpe/`: Rust-based tokenizer that must be built with `maturin develop`.
- `tasks/`: Evaluation suites (ARC, GSM8K, MMLU, etc.); useful when adding new benchmarks or debugging regressions.
- `tests/`: Currently focused on tokenizer coverage; extend here before landing risky changes.
- `dev/`, `rls_dev/`, `local_rls_experiments/`: Scratchpads and prototypes—useful references but not always production-ready.

## Quick Start
```bash
# 1. Tooling
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate

# 2. Dependencies
uv sync --extra gpu   # or --extra cpu on Apple Silicon/CPU-only boxes

# 3. Rust tokenizer (required for scripts touching tokenization)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## Common Pipelines
```bash
# End-to-end "$100 speedrun" (4h on 8xH100)
bash speedrun.sh

# Larger depth-32 (~$1000) run
bash run1000.sh

# Individual stages (torchrun for multi-GPU, drop it for single GPU)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl

# Evaluation / inference
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
python -m scripts.chat_cli -p "Hello nanochat!"
python -m scripts.chat_web  # serves the UI
```

## Agent Workflow
1. **Triage quickly.** Identify whether the request is doc-only, code change, experiment, or investigation.
2. **Plan when non-trivial.** Break work into ≤5 steps. Update the plan after completing each major step.
3. **Stay within sandbox rules.** Respect `workspace-write` boundaries; never run destructive git commands unless explicitly told.
4. **Cite edits.** In final responses, reference touched files with line numbers (`path/to/file.py:42`).
5. **Verify.** Run the lightest possible tests (unit, formatting, lint) relevant to your change; if you skip tests, state why.
6. **Leave breadcrumbs.** Mention follow-up ideas or caveats so the next agent can continue without rediscovery.

## Coding Guidelines
- Favor clarity over cleverness; mirror the repo’s minimal style (see `nanochat/gpt.py` and `scripts/base_train.py`).
- Keep dependencies stable—prefer editing existing utilities before adding new third-party packages.
- Add comments only when the intent is non-obvious (e.g., custom distributed sync, tricky tensor shapes).
- When touching training/inference hyperparameters, document rationale in code or `progress.md`.

## Testing Tips
```bash
# Tokenizer tests (fast, default smoke test)
python -m pytest tests/test_rustbpe.py -v -s

# Regenerate training report
python -m nanochat.report reset
python -m nanochat.report generate
```
- For data-oriented changes, run a small shard on CPU/MPS first.
- Capture metrics or logs in `progress.md` when experiments deviate from standard scripts.

## Troubleshooting Cheatsheet
- **OOM / CUDA errors:** reduce `--device_batch_size`; gradient accumulation adjusts automatically.
- **Tokenizer mismatch:** rerun `maturin develop` and ensure `python -m scripts.tok_eval` still works.
- **Wandb noise:** set `WANDB_RUN=dummy` when logging isn’t needed; run `wandb login` beforehand when it is.
- **Long runs:** prefer `screen` or `tmux` + logfiles (`screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh`).

## Collaboration Etiquette
- Keep responses concise—highlight findings first, then context.
- Never undo user changes without confirmation.
- When uncertain, ask clarifying questions instead of guessing.
- Suggest next steps (tests, benchmarks, docs) when handing off.

Happy hacking! Together we keep nanochat approachable, reproducible, and fun.
