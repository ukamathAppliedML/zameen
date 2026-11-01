# Models

The Colab runs wrote their trained LoRA adapters outside this repository (Google Drive, or the
Colab disk). This folder holds a **placeholder for each one**: a model card with its results and an
empty `adapter/` folder for you to fill.

| Run | Base model | New-poet CER | Where the adapter is | Put it here | Hugging Face |
|---|---|---|---|---|---|
| [qwen2.5-14b-medium](qwen2.5-14b-medium/) ★ | Qwen/Qwen2.5-14B-Instruct | 1.5% | Colab disk only ¹ | `models/qwen2.5-14b-medium/adapter/` | _TODO_ |
| [qwen2.5-14b-small](qwen2.5-14b-small/) | Qwen/Qwen2.5-14B-Instruct | 2.9% | Colab disk only ¹ | `models/qwen2.5-14b-small/adapter/` | _TODO_ |
| [sarvam-30b-medium](sarvam-30b-medium/) | sarvamai/sarvam-30b | 5.0% | `MyDrive/ghazal_tuning/sarvam-30b-medium/adapter` | `models/sarvam-30b-medium/adapter/` | _TODO_ |
| [sarvam-30b-small](sarvam-30b-small/) | sarvamai/sarvam-30b | 8.0% | `MyDrive/ghazal_tuning/sarvam-30b-small/adapter` | `models/sarvam-30b-small/adapter/` | _TODO_ |

★ recommended.

¹ **The two Qwen runs saved to the Colab disk** (`/content/ghazal_tuning/<run>/adapter`, with
`SAVE_TO_DRIVE` off), **not to Google Drive**. That disk is wiped when a Colab session ends, so
unless you downloaded the folders, these adapters need re-creating. The run is reproducible (same
seed, same splits): `zameen run -c configs/qwen2.5-14b-medium.yaml` takes about 1.5 hours on one
A100 80 GB and writes the adapter to `runs/qwen2.5-14b-medium/adapter`; copy that here. In Colab,
use the notebook with `SAVE_TO_DRIVE = True`. The two Sarvam adapters are on Drive.

The same table, machine-readable, is [`registry.yaml`](registry.yaml).

## Filling a placeholder

1. In Google Drive, open `MyDrive/ghazal_tuning/<run>/adapter` and download the folder (or take
   `runs/<run>/adapter` from a `zameen run`). It holds `adapter_config.json`,
   `adapter_model.safetensors` and tokenizer files: roughly 40 MB for Sarvam's adapter and a few
   hundred MB for Qwen2.5-14B's.
2. Copy the files into `models/<run>/adapter/`, next to the `.gitkeep`.
3. Check it: `zameen eval -c configs/<run>.yaml --adapter models/<run>/adapter` reproduces the
   numbers in `results/<run>/` (on a GPU).

The adapters are **not committed to git** (`.gitignore` excludes `models/**/adapter/*`): binaries
bloat the repository forever. Publish them to the Hugging Face Hub instead and reference them by
id; every command that takes `--adapter` accepts a Hub id as well as a folder.

```bash
pip install -e '.[hub]' && huggingface-cli login
python scripts/push_to_hub.py models/qwen2.5-14b-medium udaykamath/zameen-qwen2.5-14b-medium
python scripts/model_card.py results/qwen2.5-14b-medium --hf-repo udaykamath/zameen-qwen2.5-14b-medium
```

Then replace the _TODO_ above and `hf_repo: null` in `registry.yaml`.

## What else is on Drive

Each `MyDrive/ghazal_tuning/<run>/` folder also has `predictions_base.jsonl` and
`predictions_tuned.jsonl` (every test prompt with the model's raw output and its scores),
`summary.json`, `validation_curve.csv` and the figures. To keep the predictions with the results,
copy them into `results/<run>/`; they are what the planned error audit (how much remaining error is
inconsistency in Rekhta's own text) needs.
