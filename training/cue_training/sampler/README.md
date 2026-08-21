# Cue embedding diffusion sampler

Conditional latent diffusion over the frozen encoder bottleneck (1024-d).

## Commands

```bash
# 1) Build the prior bank, FAISS index, mu/sigma, bank_report.json
#    Prior = every session under prior_data_root, i.e. what `data pull` writes.
#    prior_holdout_frac splits off prior_dev; persona manuals are never read.
cue sampler-build-bank --config configs/sampler.json

# 2) Train sampler (reads bank_dir; writes output/best and output/last)
cue sampler-train --config configs/sampler.json --no_wandb

# 3) Sample embeddings (optional condition file of raw cue vectors)
cue sampler-sample --checkpoint /path/to/output/best --n 8 --output samples.json
cue sampler-sample --checkpoint /path/to/output/best --condition cond.json --guidance_w 1.5
```

Build-bank and train both detect `WORLD_SIZE` and run DDP; the CLI does not auto-relaunch
(unlike `cue model refine`). Encode progress uses `prior_data_root/streaming_index.json`
when present (written by `data pull`) so the tqdm bar has a total.

```bash
# Multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
uv run torchrun --standalone --nproc_per_node=4 -m cue_training.cli sampler build-bank --config configs/sampler.json
uv run torchrun --standalone --nproc_per_node=4 -m cue_training.cli sampler train --config configs/sampler.json --no_wandb
```

## Artifacts

| Path | Contents |
|------|----------|
| `bank_dir/prior_train.npy` (+ `prior_dev`) | Memmapped embeddings; training and validation read only these |
| `bank_dir/prior.faiss` | Cosine (IP) index over `prior_train` |
| `bank_dir/mu.npy`, `sigma.npy` | Prior-train standardization |
| `bank_dir/bank_report.json` | Sanity stats + real train-vs-dev baselines |
| `output/best/sampler.pt` | Live + EMA weights, schedule, mu/sigma, LN affine |

## Source mixture

`source_balance_alpha` sets how much anchor mass each corpus gets: `n_source ** alpha`,
normalized. `1.0` is uniform over rows, so the mixture is whatever the bank holds -- which the
annotation `per_dataset_cap` makes an artifact of that cap rather than of how common
those users are. `0.0` gives every corpus equal mass; values in between interpolate.
Check the current mixture in `bank_report.json` under `shards.prior_train.source_counts`,
and the resulting mass in the `anchor mass by source` line at the start of training.

`condition_same_source_frac` controls the conditioning sets, which are otherwise plain
kNN over the whole bank and can mix corpora. That fraction of episodes drops
other-source neighbors before choosing the set, so the model sees coherent single-corpus
sets like the ones inference conditions on; the rest stay mixed, which is what teaches it
to handle sets sitting between corpora. `0.0` (default) is the original all-kNN behavior.
Restricted episodes search a deeper candidate list so the set can still be filled, and
fall back to the mixed neighbors when a row has no same-source neighbor nearby.

## Resume

```bash
cue sampler-train --config configs/sampler.json --resume /path/to/output/last
```

## Conditional sampling

Pass raw (unstandardized) cue embeddings as JSON `[K,D]` / `[D]` or JSONL rows with `embedding` / `trajectory_embedding`. The sampler standardizes with checkpoint `mu`/`sigma` and applies CFG with `guidance_w` (0 or no set → unconditional).
