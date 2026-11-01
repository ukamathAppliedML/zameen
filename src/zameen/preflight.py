"""Checks before the expensive part of a run.

1. Reasoning. Does the model think before answering? If so (and ``think_mode`` is auto), an empty,
   closed think block is pre-filled for every prompt, base and tuned alike. If the model keeps
   reasoning anyway, reasoning is allowed with ``think_tokens`` extra tokens and only the text after
   ``</think>`` is scored.
2. Sanity gate. Does the model produce language at all (knows Paris, writes Devanagari when asked,
   no degenerate repetition)? If not, retry without the KV cache (broken caches are common in
   custom model code), then stop with a clear error instead of an hour of meaningless baseline.
3. Timing. One batch is timed to estimate each evaluation pass.
"""

from __future__ import annotations

import math
import re
import time

from .metrics import score
from .tasks import make_example
from .text import SCRIPTS, scripts_of, strip_think

_REASONING = re.compile(r"^\s*(<think>|\d+\.\s+\*\*Analy[sz]e|\*\*Analy[sz]e|Okay, |Let me |First, )", re.IGNORECASE)


class PreflightError(RuntimeError):
    pass


def reasons(text: str) -> bool:
    return "<think>" in text or "</think>" in text or bool(_REASONING.match(text))


def probe_poem(poems: list[dict]) -> dict:
    full = [p for p in poems if len(scripts_of(p)) == 3]
    return next((p for p in full if p["poet"] == "mirza-ghalib"), full[0])


def detect_reasoning(backend, probe_ex: dict, cfg, log) -> tuple[bool, int]:
    backend.think_prefill, extra = False, 0
    raw_free = backend.generate([probe_ex["prompt"]], 256, show=False)[0]
    thinks = reasons(raw_free) or "<think>" in backend.render(probe_ex["prompt"])[-40:]
    if cfg.think_mode == "prefill" or (cfg.think_mode == "auto" and thinks):
        backend.think_prefill = True
        raw_pre = backend.generate([probe_ex["prompt"]], cfg.answer_tokens, show=False)[0]
        if reasons(raw_pre):
            log(f"⚠️  The model keeps reasoning with an empty think block pre-filled → allowing reasoning "
                f"(+{cfg.think_tokens} tokens per answer); only the text after </think> is scored.")
            backend.think_prefill, extra = False, cfg.think_tokens
    elif cfg.think_mode == "allow" and thinks:
        extra = cfg.think_tokens
    log(f"Model reasons before answering: {thinks} → think prefill {'ON' if backend.think_prefill else 'OFF'}"
        + (f", reasoning allowed (+{extra} tokens)" if extra else ""))
    return thinks, extra


def sanity(backend, poem: dict, extra: int) -> tuple[bool, dict, str, str]:
    budget = 512 + extra
    a = backend.generate(["What is the capital of France? Answer with one word."], budget, show=False)[0]
    pe = make_example(poem, "roman", "devanagari", 0)       # the easiest direction: nearly mechanical
    b = backend.generate([pe["prompt"]], budget, show=False)[0]
    ans, unfinished = strip_think(b)
    if unfinished or (not ans and reasons(b)):
        ans = ""
    words = ans.split()
    letters = [c for c in ans if c.isalpha() or "ऀ" <= c <= "ॿ"]
    deva = sum("ऀ" <= c <= "ॿ" for c in ans)
    checks = {"knows 'Paris'": "paris" in a.lower(),
              "writes Devanagari": bool(letters) and deva / len(letters) > 0.3,
              "not degenerate repetition": len(words) > 3 and len(set(words)) / len(words) > 0.3}
    return all(checks.values()), checks, a, b


def run_preflight(backend, poems: list[dict], data, cfg, log) -> dict:
    poem = probe_poem(poems)
    sample = [p for p in poems if len(scripts_of(p)) == 3][:150]
    tok_stats = backend.tokenizer_stats({s: [l for p in sample for l in p["text"][s]] for s in SCRIPTS})
    if tok_stats:
        log("Tokenizer: " + " · ".join(f"{s} {v['tokens/word']} pieces/word" for s, v in tok_stats.items()))

    probe_ex = make_example(poem, "nastaliq", "devanagari", 0)
    _thinks, extra = detect_reasoning(backend, probe_ex, cfg, log)
    ok, checks, a, b = sanity(backend, poem, extra)
    log(f"Sanity: {checks}")
    if not ok and reasons(b) and not extra:
        log(f"The model is reasoning, not broken: it ran out of tokens. Allowing reasoning (+{cfg.think_tokens}).")
        backend.think_prefill, extra = False, cfg.think_tokens
        ok, checks, a, b = sanity(backend, poem, extra)
        log(f"Sanity with reasoning allowed: {checks}")
    if not ok and not reasons(b):
        log("Output looks broken. Retrying without the KV cache (a common bug in custom model code)...")
        backend.use_cache = False
        ok, checks, a, b = sanity(backend, poem, extra)
        if ok:
            log("✅ Works without the KV cache: slower but correct.")
            _thinks, extra = detect_reasoning(backend, probe_ex, cfg, log)   # first check saw broken output
            ok, checks, a, b = sanity(backend, poem, extra)
    if not ok:
        env = backend.environment()
        raise PreflightError(
            ("The model reasons at length and never reaches an answer; try think_tokens: 2048.\n" if reasons(b)
             else "The model's output is degenerate, so a baseline would be meaningless. Stopping.\n")
            + f"Environment: {env}\nSample output: {b[-300:]!r}\n"
            "Fixes: for Sarvam-30B use transformers 4.x (pip install -e '.[sarvam]'); "
            "or try model: qwen2.5-7b to validate the pipeline.")
    log(f"✅ Model output is sane: {strip_think(b)[0][:120]!r}")

    s = score(probe_ex, backend.generate([probe_ex["prompt"]], cfg.answer_tokens + extra, show=False)[0])
    log(f"Probe (Ghalib, Nastaliq → Devanagari) CER {s['char_err'] / max(s['chars'], 1):.1%}")

    batch = data.test["test_seen"][:backend.gen_batch]
    t0 = time.time()
    backend.generate([e["prompt"] for e in batch], cfg.answer_tokens + extra, show=False)
    sec = time.time() - t0
    n_eval = sum(len(v) for v in data.test.values())
    est = sec * math.ceil(n_eval / max(backend.gen_batch, 1)) * 1.1
    log(f"One batch of {len(batch)}: {sec:.1f}s → each evaluation ({n_eval} prompts) ≈ {est / 60:.0f} min")
    return {"think_prefill": backend.think_prefill, "extra_tokens": extra, "use_cache": backend.use_cache,
            "sanity": checks, "tokenizer": tok_stats, "est_eval_min": round(est / 60, 1)}
