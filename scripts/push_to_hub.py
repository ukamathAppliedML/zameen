"""Publish an adapter folder (and its model card) to the Hugging Face Hub.

    pip install -e '.[hub]'
    huggingface-cli login
    python scripts/push_to_hub.py models/qwen2.5-14b-medium udaykamath/zameen-qwen2.5-14b-medium

Then set hf_repo in models/registry.yaml and regenerate the card:
    python scripts/model_card.py results/qwen2.5-14b-medium --hf-repo udaykamath/zameen-qwen2.5-14b-medium
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="models/<run> (with adapter/ filled in and README.md)")
    ap.add_argument("repo_id", help="e.g. udaykamath/zameen-qwen2.5-14b-medium")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    src = Path(a.model_dir)
    if not (src / "adapter" / "adapter_config.json").exists():
        raise SystemExit(f"{src}/adapter is still a placeholder: copy the adapter from Google Drive first.")
    with tempfile.TemporaryDirectory() as tmp:
        for f in (src / "adapter").iterdir():
            if f.name != ".gitkeep":
                shutil.copy2(f, tmp)
        shutil.copy2(src / "README.md", Path(tmp) / "README.md")
        api = HfApi()
        api.create_repo(a.repo_id, private=a.private, exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=a.repo_id)
    print(f"https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
