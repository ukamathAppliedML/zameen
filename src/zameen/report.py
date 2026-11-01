"""Run summaries, Markdown reports, figures and cross-run comparison.

Every run (from the pipeline or imported from a Colab notebook) is described by one
``summary.json`` with the same schema, so ``zameen compare`` works across both:

    splits.<split>.directions.<"Nastaliq → Devanagari" | "all directions">.{before,after}.<metric>
    splits.<split>.phenomena.<target script>.<feature>.{before,after}
    splits.<split>.form.{radif_kept,qafiya_kept}.{before,after}
    validation_curve: [{examples_seen, val_CER, val_exact, "from <script>": ...}]
    training: {train_examples, trainable_params, best_val_step, ...}
    timing_minutes: {stage: minutes}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .metrics import direction_table, form_rates, mark, phenomena_table
from .splits import TEST_SPLITS
from .tasks import DIR_LABEL

SCHEMA = 1
DIR_ORDER = list(DIR_LABEL.values()) + ["all directions"]
SPLIT_TITLE = {"test_seen": "Familiar poets (test_seen)", "test_unseen": "Poets never seen in training (test_unseen)"}


# ---------------------------------------------------------------------------- building summaries
def _directions(base_rows, tuned_rows, split) -> dict:
    out: dict = {}
    for tag, rows in (("before", base_rows), ("after", tuned_rows)):
        if not rows:
            continue
        t = direction_table(rows, split)
        for d, rec in t.to_dict("index").items():
            out.setdefault(d, {})[tag] = {k: float(v) for k, v in rec.items()}
    return out


def _phenomena(base_rows, tuned_rows, split) -> dict:
    out: dict = defaultdict(dict)
    b = phenomena_table(base_rows, split) if base_rows else {}
    t = phenomena_table(tuned_rows, split) if tuned_rows else {}
    for key in sorted(set(b) | set(t)):
        out[key[0]][key[1]] = {"before": b.get(key), "after": t.get(key)}
    return dict(out)


def _form(base_rows, tuned_rows, split) -> dict:
    b = form_rates(base_rows, split) if base_rows else {}
    t = form_rates(tuned_rows, split) if tuned_rows else {}
    return {k: {"before": b.get(k), "after": t.get(k)} for k in ("radif_kept", "qafiya_kept")}


def build_summary(cfg, base_rows, tuned_rows, train_res=None, environment=None, preflight=None,
                  data_stats=None, timing=None) -> dict:
    return {
        "schema": SCHEMA, "source": "pipeline", "run": cfg.name, "model": cfg.spec["id"], "model_key": cfg.model,
        "size": cfg.size, "preset": cfg.preset, "environment": environment or {}, "preflight": preflight or {},
        "data": data_stats or {},
        "training": None if train_res is None else {
            "train_examples": train_res.train_examples, "trainable_params": train_res.trainable_params,
            "best_val_step": train_res.best_step, "lora_r": cfg.lora_r, "learning_rate": cfg.learning_rate,
            "resumed_from": train_res.resumed_from, "reused": train_res.reused},
        "splits": {s: {"directions": _directions(base_rows, tuned_rows, s),
                       "phenomena": _phenomena(base_rows, tuned_rows, s),
                       "form": _form(base_rows, tuned_rows, s)} for s in TEST_SPLITS},
        "validation_curve": [] if train_res is None else train_res.curve,
        "timing_minutes": timing or {},
    }


# ---------------------------------------------------------------------------- rendering
def pct(x, digits=1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def headline(summary: dict, split: str = "test_unseen") -> dict:
    d = summary["splits"].get(split, {}).get("directions", {}).get("all directions", {})
    return {"cer_before": d.get("before", {}).get("CER"), "cer_after": d.get("after", {}).get("CER"),
            "exact_before": d.get("before", {}).get("exact"), "exact_after": d.get("after", {}).get("exact")}


def render_markdown(summary: dict) -> str:
    L = [f"# {summary['run']}: {summary['model']}", ""]
    tr = summary.get("training") or {}
    meta = [f"size **{summary.get('size')}**"]
    if tr.get("train_examples"):
        meta.append(f"{tr['train_examples']:,} training examples")
    if tr.get("trainable_params"):
        meta.append(f"{tr['trainable_params'] / 1e6:.1f} M trainable LoRA parameters")
    if tr.get("best_val_step") is not None:
        meta.append(f"best validation checkpoint at step {tr['best_val_step']}")
    env = summary.get("environment") or {}
    if env.get("gpu"):
        q = env.get("quantization", "")
        meta.append(f"{env['gpu']}, {'bf16' if q == 'none' else q}")
    L += [" · ".join(meta), "", f"Source: {summary.get('source')}.", ""]
    for split in TEST_SPLITS:
        sp = summary["splits"].get(split)
        if not sp or not sp["directions"]:
            continue
        L += [f"## {SPLIT_TITLE[split]}", "", "| Direction | CER | CER loose | WER | Exact shers |", "|---|---|---|---|---|"]
        for d in DIR_ORDER:
            rec = sp["directions"].get(d)
            if not rec:
                continue
            cells = []
            for m in ("CER", "CER loose", "WER", "exact"):
                b, a = rec.get("before", {}).get(m), rec.get("after", {}).get(m)
                cells.append(f"{pct(b)} → **{pct(a)}**" if a is not None else pct(b))
            name = f"**{d}**" if d == "all directions" else d
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
        form = sp.get("form") or {}
        if any(v.get("after") is not None for v in form.values()):
            L += ["Form kept (rule-based check on matlas): " + " · ".join(
                f"{k.replace('_kept', '')} {pct(v.get('before'))} → **{pct(v.get('after'))}**"
                for k, v in form.items() if v.get("after") is not None), ""]
        ph = sp.get("phenomena") or {}
        if ph:
            L += ["<details><summary>Word accuracy by phenomenon</summary>", "",
                  "| Target script | Words with | Before | After |", "|---|---|---|---|"]
            for script, feats in ph.items():
                for f, v in feats.items():
                    L.append(f"| {script} | {f} | {pct(v.get('before'))} | {pct(v.get('after'))} |")
            L += ["", "</details>", ""]
    curve = summary.get("validation_curve") or []
    if curve:
        L += ["## Validation curve", "", "| Examples seen | Validation CER | Validation exact |", "|---|---|---|"]
        L += [f"| {int(r['examples_seen']):,} | {pct(r['val_CER'])} | {pct(r.get('val_exact'), 0)} |" for r in curve]
        L.append("")
    timing = summary.get("timing_minutes") or {}
    if timing:
        L += ["## Time", "", "| Stage | Minutes |", "|---|---|"]
        L += [f"| {k} | {v:.1f} |" for k, v in timing.items()]
        L += [f"| **total** | **{sum(timing.values()):.0f}** |", ""]
    return "\n".join(L)


def render_examples(base_rows, tuned_rows, per_direction: int = 2) -> str:
    bmap = {(r["split"], r["id"]): r for r in base_rows}
    shown: dict = defaultdict(int)
    L = ["# Side by side: Rekhta, before, after", "",
         "Words that differ from Rekhta's text are in ⟦ ⟧. Poets never seen in training, matla only.", ""]
    for r in tuned_rows:
        d = (r["src"], r["tgt"])
        if r["split"] != "test_unseen" or r["sher"] != 0 or shown[d] >= per_direction:
            continue
        shown[d] += 1
        b = bmap.get((r["split"], r["id"]), {})
        L += [f"### {DIR_LABEL[d]} · {r['poet']}", "",
              f"- **Rekhta:** {r['gold'].replace(chr(10), ' / ')}",
              f"- **before:** {mark(r['gold'], b.get('pred', '')) or '(no answer)'}",
              f"- **after:** {mark(r['gold'], r['pred']) or '(no answer)'}", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------- figures
def plot_learning_curve(summary: dict, path: Path) -> Path | None:
    curve = summary.get("validation_curve") or []
    if not curve:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["examples_seen"] for r in curve]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k in [k for k in curve[0] if k.startswith("from ")]:
        ax.plot(x, [r[k] for r in curve], marker="o", lw=1.5, label=k)
    ax.plot(x, [r["val_CER"] for r in curve], "k--", lw=2.2, label="all")
    ax.set(xlabel="training examples seen", ylabel="validation CER (lower is better)",
           title=f"{summary['run']}: validation error while training")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_before_after(summary: dict, path: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    for ax, split in zip(axes, TEST_SPLITS):
        dirs = summary["splits"].get(split, {}).get("directions", {})
        names = [d for d in DIR_ORDER if d in dirs]
        b = [dirs[d].get("before", {}).get("CER", 0) for d in names]
        a = [dirs[d].get("after", {}).get("CER", 0) for d in names]
        xs = range(len(names))
        ax.bar([i - 0.2 for i in xs], b, 0.4, label="before", color="#B8C0CC")
        ax.bar([i + 0.2 for i in xs], a, 0.4, label="after", color="#2F6F73")
        ax.set_xticks(list(xs), names, rotation=30, ha="right", fontsize=8)
        ax.set_title(SPLIT_TITLE[split], fontsize=10)
    axes[0].set_ylabel("character error rate (lower is better)")
    axes[0].legend()
    fig.suptitle(f"{summary['run']}: before vs after fine-tuning")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_run_reports(summary: dict, out_dir: str | Path, base_rows=None, tuned_rows=None) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (out / "results.md").write_text(render_markdown(summary), encoding="utf-8")
    files = [out / "summary.json", out / "results.md"]
    if summary.get("validation_curve"):
        import pandas as pd
        pd.DataFrame(summary["validation_curve"]).to_csv(out / "validation_curve.csv", index=False)
        files.append(out / "validation_curve.csv")
    for p in (plot_learning_curve(summary, out / "learning_curve.png"),
              plot_before_after(summary, out / "before_after.png")):
        if p:
            files.append(p)
    if base_rows and tuned_rows:
        (out / "examples.md").write_text(render_examples(base_rows, tuned_rows), encoding="utf-8")
        files.append(out / "examples.md")
    return files


# ---------------------------------------------------------------------------- comparison
def load_summary(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


def compare(summaries: list[dict], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    L = ["# Comparison", "", "Familiar poets / poets never seen in training. CER = characters wrong (lower is better).", "",
         "| Run | Training examples | CER before | CER after | Exact shers after | Radif kept | Qafiya kept | Training min |",
         "|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        def pair(metric, tag, s=s):
            v = [s["splits"].get(sp, {}).get("directions", {}).get("all directions", {}).get(tag, {}).get(metric)
                 for sp in TEST_SPLITS]
            return " / ".join(pct(x) for x in v)

        def form(k, s=s):
            return " / ".join(pct(s["splits"].get(sp, {}).get("form", {}).get(k, {}).get("after"), 0) for sp in TEST_SPLITS)
        tr = s.get("training") or {}
        L.append(f"| {s['run']} | {tr.get('train_examples', '–'):,} | {pair('CER', 'before')} | **{pair('CER', 'after')}** | "
                 f"{pair('exact', 'after')} | {form('radif_kept')} | {form('qafiya_kept')} | "
                 f"{s.get('timing_minutes', {}).get('training', float('nan')):.0f} |")
    L += ["", "## After tuning, by direction (poets never seen in training)", "",
          "| Direction | " + " | ".join(s["run"] for s in summaries) + " |", "|---|" + "---|" * len(summaries)]
    for d in DIR_ORDER:
        L.append(f"| {d} | " + " | ".join(
            pct(s["splits"].get("test_unseen", {}).get("directions", {}).get(d, {}).get("after", {}).get("CER"))
            for s in summaries) + " |")
    L.append("")
    bars = _plot_directions(summaries, out / "comparison_directions.png")
    if bars:
        L += [f"![after tuning by direction]({bars.name})", ""]
    fig_path = _plot_curves(summaries, out / "comparison_curves.png")
    if fig_path:
        L += ["## Validation curves", "", f"![validation curves]({fig_path.name})", ""]
    (out / "comparison.md").write_text("\n".join(L), encoding="utf-8")
    return out / "comparison.md"


def _plot_curves(summaries, path: Path) -> Path | None:
    if not any(s.get("validation_curve") for s in summaries):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    family = {"qwen": "#2F6F73", "sarvam": "#C27C2C"}
    for s in summaries:
        c = [r for r in s.get("validation_curve") or [] if r["examples_seen"] > 0]
        if c:
            color = next((v for k, v in family.items() if k in s["run"].lower()), None)
            style = "-" if (s.get("training") or {}).get("train_examples", 0) >= 10000 else "--"
            ax.plot([r["examples_seen"] for r in c], [r["val_CER"] for r in c], style, marker="o", color=color,
                    label=s["run"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(xlabel="training examples seen", ylabel="validation CER", title="Validation error while training")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_directions(summaries, path: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    family = {"qwen": ["#9CC9C9", "#2F6F73"], "sarvam": ["#E3C99A", "#C27C2C"]}
    seen: dict = {}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.8 / max(len(summaries), 1)
    for i, s in enumerate(summaries):
        dirs = s["splits"].get("test_unseen", {}).get("directions", {})
        vals = [dirs.get(d, {}).get("after", {}).get("CER") or 0 for d in DIR_ORDER]
        fam = next((k for k in family if k in s["run"].lower()), None)
        color = None
        if fam:
            color = family[fam][min(seen.get(fam, 0), 1)]
            seen[fam] = seen.get(fam, 0) + 1
        ax.bar([j - 0.4 + w * (i + 0.5) for j in range(len(DIR_ORDER))], vals, w, label=s["run"], color=color)
    ax.set_xticks(range(len(DIR_ORDER)), DIR_ORDER, rotation=20, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(ylabel="characters wrong after tuning", title="Poets never seen in training: error by direction (lower is better)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
