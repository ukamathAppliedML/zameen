"""Command-line interface: ``zameen <command> --help`` for details."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from . import __version__
from .config import MODELS, PRESETS, RunConfig


def _cfg(args) -> RunConfig:
    return RunConfig.from_yaml(
        getattr(args, "config", None), model=getattr(args, "model", None), size=getattr(args, "size", None),
        n_train=getattr(args, "n_train", None), output_dir=getattr(args, "output_dir", None),
        corpus=getattr(args, "corpus", None), run_name=getattr(args, "run_name", None))


def _add_cfg_args(p, model=True):
    p.add_argument("-c", "--config", help="YAML config (see configs/)")
    if model:
        p.add_argument("--model", choices=sorted(MODELS), help="override the config's model")
    p.add_argument("--size", choices=sorted(PRESETS), help="override the config's size preset")
    p.add_argument("--n-train", type=int, help="override the number of training examples")
    p.add_argument("--corpus", help="JSONL export or Rekhta-style folder (default: public sample)")
    p.add_argument("--output-dir", help="where run folders go (default: runs)")
    p.add_argument("--run-name", help="run folder name (default: <model>-<size>)")


def cmd_data(args):
    from .pipeline import export_datasets, prepare
    from .progress import Progress
    cfg = _cfg(args)
    _poems, data = prepare(cfg, Progress())
    print(json.dumps(data.stats(), ensure_ascii=False, indent=2))
    if args.export:
        files = export_datasets(data, Path(args.export))
        print("Wrote:", *[str(f) for f in files], sep="\n  ")


def cmd_run(args):
    from .pipeline import run
    run(_cfg(args), backend_kind=args.backend, skip_baseline=args.skip_baseline)


def cmd_eval(args):
    from .pipeline import evaluate_adapter
    adapter = None if args.adapter in (None, "none") else args.adapter
    evaluate_adapter(_cfg(args), adapter, backend_kind=args.backend, tag="base" if adapter is None else args.tag)


def cmd_transliterate(args):
    from .pipeline import transliterate
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    lines = [l.strip() for l in text.replace(" / ", "\n").splitlines() if l.strip()]
    print(transliterate(_cfg(args), args.adapter, lines, args.to, backend_kind=args.backend))


def cmd_compare(args):
    from .report import compare, load_summary
    summaries = [load_summary(p) for p in args.runs]
    out = compare(summaries, args.out)
    print(out.read_text(encoding="utf-8"))


def cmd_import(args):
    from .importer import import_notebook
    s = import_notebook(args.notebook, args.out, args.run_name)
    d = s["splits"]["test_unseen"]["directions"].get("all directions", {})
    print(f"Imported {s['run']} → {args.out}: unseen-poet CER "
          f"{d.get('before', {}).get('CER')} → {d.get('after', {}).get('CER')}")


def cmd_smoke(args):
    """The whole pipeline on CPU in seconds, with the oracle backend and a bundled mini corpus."""
    from .pipeline import run
    corpus = args.corpus or str(Path(__file__).resolve().parent / "data" / "mini_corpus.jsonl")
    cfg = RunConfig.from_yaml(None, model="qwen2.5-7b", size="smoke", corpus=corpus,
                              output_dir=args.output_dir or "runs", run_name="smoke", test_frac=0.3, val_frac=0.15)
    s = run(cfg, backend_kind="oracle")
    ok = all(c["ok"] for c in s.get("checks", []))
    print("\nSMOKE OK" if ok else "\nSMOKE: some checks failed")
    sys.exit(0 if ok else 1)


def cmd_doctor(args):
    print(f"zameen {__version__} · Python {platform.python_version()} · {platform.platform()}")
    try:
        import torch
        print(f"torch {torch.__version__} · CUDA {'yes: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no'}")
        if torch.cuda.is_available():
            print(f"GPU memory {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    except ImportError:
        print("torch not installed: CPU-only commands work (data, compare, import, smoke). "
              "For training: pip install -e '.[train]'")
        return
    try:
        import transformers
        print(f"transformers {transformers.__version__}")
        from .backends.hf import check_versions
        cfg = RunConfig(model=args.model or "qwen2.5-14b")
        problems = check_versions(cfg)
        print("\n".join("⚠️  " + p for p in problems) or f"✅ stack looks fine for {cfg.model}")
    except ImportError:
        print("transformers not installed: pip install -e '.[train]'")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zameen", description="Ghazal transliteration across Nastaliq, Devanagari and Roman.")
    ap.add_argument("--version", action="version", version=f"zameen {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("data", help="build the datasets, print statistics, optionally export them")
    _add_cfg_args(p)
    p.add_argument("--export", help="folder to write train / val / test JSONL and split membership")
    p.set_defaults(fn=cmd_data)

    p = sub.add_parser("run", help="full experiment: baseline, LoRA fine-tuning, evaluation, reports")
    _add_cfg_args(p)
    p.add_argument("--backend", default="hf", choices=["hf", "oracle"])
    p.add_argument("--skip-baseline", action="store_true", help="skip the before-tuning evaluation")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("eval", help="evaluate a saved adapter (or the base model with --adapter none)")
    _add_cfg_args(p)
    p.add_argument("--adapter", required=True, help="adapter folder (e.g. models/qwen2.5-14b-medium/adapter), "
                                                     "a Hugging Face repo id, or 'none' for the base model")
    p.add_argument("--tag", default="tuned", help="name of the predictions file (predictions_<tag>.jsonl)")
    p.add_argument("--backend", default="hf", choices=["hf", "oracle"])
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("transliterate", help="transliterate a sher with a tuned adapter")
    _add_cfg_args(p)
    p.add_argument("--adapter", help="adapter folder or Hugging Face repo id")
    p.add_argument("--to", required=True, choices=["nastaliq", "devanagari", "roman"])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="the sher; separate the two lines with a newline or ' / '")
    g.add_argument("--file", help="a text file holding the sher")
    p.add_argument("--backend", default="hf", choices=["hf", "oracle"])
    p.set_defaults(fn=cmd_transliterate)

    p = sub.add_parser("compare", help="compare runs side by side (folders or summary.json files)")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="results")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("import-notebook", help="turn an executed Colab notebook into a results folder")
    p.add_argument("notebook")
    p.add_argument("--out", required=True)
    p.add_argument("--run-name")
    p.set_defaults(fn=cmd_import)

    p = sub.add_parser("smoke", help="run the whole pipeline on CPU with a stand-in model (seconds)")
    p.add_argument("--corpus")
    p.add_argument("--output-dir")
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("doctor", help="check Python, GPU and library versions")
    p.add_argument("--model", choices=sorted(MODELS))
    p.set_defaults(fn=cmd_doctor)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    main()
