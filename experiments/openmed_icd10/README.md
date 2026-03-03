# OpenMed ICD-10

Trains `Qwen3-1.7B` to assign ICD-10 diagnostic codes to clinical notes using LoRA RL against the [`maziyar/openmed_icd10`](https://app.primeintellect.ai/dashboard/environments/maziyar/openmed_icd10) environment.

> Note: This example uses 4 GPUs on a single node: 2 for training (FSDP) and 2 for inference (data parallel).

> Note: `attn = "flash_attention_3"` requires H100s. Switch to `"flash_attention_2"` for other GPU types.

## Setup

Install the environment:

```bash
prime env install maziyar/openmed_icd10
```

Verify it's installed:

```bash
uv run python -c "import openmed_icd10"
```

## RL (4 GPUs: 2 train + 2 inference)

Submit a 4-GPU RL job:

```bash
medarc_slurm rl examples/openmed_icd10/rl.toml \
    --output-dir output/examples/openmed-icd10 \
    --train-gpus 2 \
    --infer-gpus 2 \
    --auto-auth \
    --wandb.project openmed-icd10 --wandb.name openmed-icd10-1.7b-lora
```

Or preview without submitting:

```bash
medarc_slurm rl examples/openmed_icd10/rl.toml \
    --output-dir output/examples/openmed-icd10 \
    --train-gpus 2 \
    --infer-gpus 2 \
    --auto-auth \
    --dry-run \
    --wandb.project openmed-icd10 --wandb.name openmed-icd10-1.7b-lora
```

Or run locally:

```bash
medarc_train rl examples/openmed_icd10/rl.toml \
    --output-dir output/examples/openmed-icd10 \
    --train-gpus 2 \
    --infer-gpus 2 \
    --wandb.project openmed-icd10 --wandb.name openmed-icd10-1.7b-lora
```
