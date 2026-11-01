"""End-to-end runs: data → model → preflight → baseline → LoRA → tuned evaluation → reports."""

from __future__ import annotations

import json
from pathlib import Path

from .backends import load_backend
from .config import RunConfig
from .corpus import load_corpus
from .evaluate import run_eval
from .metrics import extract, score, summarize
from .preflight import probe_poem, run_preflight
from .progress import Progress
from .report import build_summary, headline, pct, write_run_reports
from .splits import Datasets, build_datasets
from .tasks import DIRECTIONS, build_prompt, make_example
from .text import detect_script


def prepare(cfg: RunConfig, prog: Progress) -> tuple[list[dict], Datasets]:
    prog.start("corpus", cfg.corpus or "public Rekhta sample")
    poems, n_all = load_corpus(cfg.corpus, cfg.cache_dir)
    prog.done("corpus", poems=len(poems), poets=len({p["poet"] for p in poems}), skipped=n_all - len(poems))
    prog.start("datasets")
    data = build_datasets(poems, cfg)
    st = data.stats()["examples"]
    prog.done("datasets", **st)
    return poems, data


def export_datasets(data: Datasets, out_dir: Path) -> list[Path]:
    """Write the exact examples and split membership, for audit or for training elsewhere."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, rows in [("train", data.train), ("val_probe", data.probe), *data.test.items()]:
        p = out_dir / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in rows)
        files.append(p)
    p = out_dir / "splits.json"
    p.write_text(json.dumps({s: sorted(x["id"] for x in ps) for s, ps in data.by_split.items()},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    files.append(p)
    return files


def _oracle_examples(poems, data) -> list[dict]:
    g = probe_poem(poems)
    extra = [e for s, t in DIRECTIONS if (e := make_example(g, s, t, 0))]
    return data.train + data.probe + [e for v in data.test.values() for e in v] + extra


def checks(base_rows, tuned_rows, curve) -> list[tuple[bool, str]]:
    from .metrics import direction_table, form_rates
    out = []
    for split in ("test_seen", "test_unseen"):
        b, t = direction_table(base_rows, split), direction_table(tuned_rows, split)
        ab, at = b.loc["all directions"], t.loc["all directions"]
        out.append((at.CER < ab.CER, f"{split}: CER {pct(ab.CER)} → {pct(at.CER)}, exact shers {pct(ab.exact)} → {pct(at.exact)}"))
        worse = [d for d in b.index[:-1] if t.loc[d, "CER"] > b.loc[d, "CER"] + 0.01]
        out.append((not worse, f"{split}: every direction improves" + (f" (worse: {', '.join(worse)})" if worse else "")))
    na = sum(r["no_answer"] for r in tuned_rows) / len(tuned_rows)
    out.append((na < 0.02, f"tuned model always answers: no-answer rate {pct(na)}"))
    ws = sum(r["wrong_script"] for r in tuned_rows) / len(tuned_rows)
    out.append((ws < 0.02, f"tuned model answers in the requested script: wrong-script rate {pct(ws)}"))
    if curve:
        first, best = curve[0]["val_CER"], min(r["val_CER"] for r in curve)
        out.append((best < first, f"validation error falls with training: {pct(first)} → best {pct(best)}"))
    fr = form_rates(tuned_rows, "test_unseen").get("radif_kept")
    if fr is not None:
        out.append((fr > 0.8, f"unseen poets: radif preserved in {pct(fr, 0)} of transliterated matlas"))
    return out


def run(cfg: RunConfig, backend_kind: str = "hf", show: bool = True, skip_baseline: bool = False) -> dict:
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.yaml")
    prog = Progress(out / "progress.jsonl", echo=show)
    log = prog.log
    log(f"Run {cfg.name} · {cfg.spec['id']} · {cfg.preset} → {out}")

    poems, data = prepare(cfg, prog)

    prog.start("load model", cfg.spec["id"])
    backend = load_backend(backend_kind, cfg, examples=_oracle_examples(poems, data) if backend_kind == "oracle" else None)
    env = backend.environment()
    prog.done("load model", **{k: v for k, v in env.items() if k in ("gpu", "quantization")})

    prog.start("preflight")
    pre = run_preflight(backend, poems, data, cfg, log)
    prog.done("preflight", think_prefill=pre["think_prefill"], est_eval_min=pre["est_eval_min"])
    max_new = cfg.answer_tokens + pre["extra_tokens"]

    base_rows: list[dict] = []
    if not skip_baseline:
        prog.start("baseline eval", f"{sum(len(v) for v in data.test.values())} prompts")
        base_rows = run_eval(backend, data.test, out / "predictions_base.jsonl", max_new, show, log)
        prog.done("baseline eval")

    def probe():
        raws = backend.generate([e["prompt"] for e in data.probe], max_new, show=False)
        rows = [dict(score(e, r), src=e["src"], tgt=e["tgt"], split="val") for e, r in zip(data.probe, raws)]
        by_src = summarize(rows, by=("src",))["CER"].to_dict()
        total = summarize(rows, by=("split",)).iloc[0]
        return float(total["CER"]), float(total["exact"]), by_src

    prog.start("training", f"{len(data.train)} examples × {cfg.preset['epochs']} epoch(s)")
    res = backend.fit(data.train, probe, cfg, out, log)
    prog.done("training", best_step=res.best_step, reused=res.reused)

    prog.start("tuned eval")
    tuned_rows = run_eval(backend, data.test, out / "predictions_tuned.jsonl", max_new, show, log)
    prog.done("tuned eval")

    summary = build_summary(cfg, base_rows, tuned_rows, res, env, pre, data.stats(), prog.minutes())
    if base_rows:
        summary["checks"] = [{"ok": ok, "check": msg} for ok, msg in checks(base_rows, tuned_rows, res.curve)]
        for c in summary["checks"]:
            log(("✅ " if c["ok"] else "⚠️  ") + c["check"])
    files = write_run_reports(summary, out, base_rows, tuned_rows)
    h = headline(summary)
    log(f"\nPoets never seen in training: CER {pct(h['cer_before'])} → {pct(h['cer_after'])}, "
        f"exact shers {pct(h['exact_before'])} → {pct(h['exact_after'])}")
    log("Reports: " + ", ".join(str(f) for f in files))
    return summary


def evaluate_adapter(cfg: RunConfig, adapter: str | None, backend_kind: str = "hf", tag: str = "tuned",
                     show: bool = True) -> dict:
    """Evaluate an existing adapter (e.g. one copied from Google Drive) on this config's test sets."""
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    prog = Progress(out / "progress.jsonl", echo=show)
    poems, data = prepare(cfg, prog)
    prog.start("load model", f"{cfg.spec['id']} + {adapter or 'no adapter'}")
    backend = load_backend(backend_kind, cfg, adapter=adapter,
                           examples=_oracle_examples(poems, data) if backend_kind == "oracle" else None)
    prog.done("load model")
    prog.start("preflight")
    pre = run_preflight(backend, poems, data, cfg, prog.log)
    prog.done("preflight")
    prog.start(f"{tag} eval")
    rows = run_eval(backend, data.test, out / f"predictions_{tag}.jsonl", cfg.answer_tokens + pre["extra_tokens"], show, prog.log)
    prog.done(f"{tag} eval")
    base_path = out / "predictions_base.jsonl"
    base_rows = [json.loads(l) for l in base_path.read_text(encoding="utf-8").splitlines()] if base_path.exists() and tag != "base" else []
    summary = build_summary(cfg, base_rows if tag != "base" else rows, rows if tag != "base" else [],
                            environment=backend.environment(), preflight=pre, data_stats=data.stats(), timing=prog.minutes())
    summary["source"] = f"evaluate ({tag}, adapter={adapter})"
    write_run_reports(summary, out, base_rows or None, rows if base_rows else None)
    h = headline(summary)
    prog.log(f"Poets never seen in training: CER {pct(h['cer_after'] if tag != 'base' else h['cer_before'])}")
    return summary


def transliterate(cfg: RunConfig, adapter: str | None, lines: list[str], target: str,
                  backend_kind: str = "hf") -> str:
    src = detect_script(" ".join(lines)) or "roman"
    e = {"src": src, "tgt": target, "source": lines, "prompt": build_prompt(lines, target, "en")}
    backend = load_backend(backend_kind, cfg, adapter=adapter)
    from .preflight import detect_reasoning
    _thinks, extra = detect_reasoning(backend, e, cfg, log=lambda m: None)   # same handling as in a run
    raw = backend.generate([e["prompt"]], cfg.answer_tokens + extra, show=False)[0]
    return extract(raw, e) or ""
