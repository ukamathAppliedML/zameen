"""Train / validation / test splits and the datasets built from them.

Splits are assigned per poem by a hash of the poem id and the seed, so they never shift when
poems are added. Two poets are held out entirely.

* train: the only data the model learns from
* val: scored during training to draw the learning curve and pick the best checkpoint
* test_seen: new poems by poets seen in training
* test_unseen: every poem by the held-out poets
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from .config import RunConfig
from .tasks import DIR_LABEL, DIRECTIONS, make_example
from .text import scripts_of, shers

SPLITS = ("train", "val", "test_seen", "test_unseen")
TEST_SPLITS = ("test_seen", "test_unseen")


def unit(key: str, seed: int) -> float:
    return int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / float(0xFFFFFFFFFFFF)


def assign_splits(poems: list[dict], cfg: RunConfig) -> dict[str, list[dict]]:
    by_split: dict[str, list[dict]] = defaultdict(list)
    for p in poems:
        u = unit(p["id"], cfg.seed)
        split = ("test_unseen" if p["poet"] in cfg.holdout_poets else
                 "test_seen" if u < cfg.test_frac else
                 "val" if u < cfg.test_frac + cfg.val_frac else "train")
        by_split[split].append(p)
    return {s: by_split.get(s, []) for s in SPLITS}


def eval_shers(poem: dict) -> list[int]:
    """The matla and one middle sher."""
    n = len(shers(poem, scripts_of(poem)[0]))
    return sorted({0, n // 2 if n >= 3 else n - 1})


def eval_set(poems: list[dict]) -> list[dict]:
    return [e for p in poems for i in eval_shers(p) for (s, t) in DIRECTIONS if (e := make_example(p, s, t, i))]


def limit_poems(poems: list[dict], n, seed: int) -> list[dict]:
    return sorted(poems, key=lambda p: unit("pick:" + p["id"], seed))[:n] if n else poems


@dataclass
class Datasets:
    by_split: dict[str, list[dict]]
    train: list[dict]
    train_available: int
    test: dict[str, list[dict]]
    probe: list[dict]

    def stats(self) -> dict:
        return {
            "poems": {s: len(self.by_split[s]) for s in SPLITS},
            "poets": {s: len({p["poet"] for p in self.by_split[s]}) for s in SPLITS},
            "examples": {"train": len(self.train), "train_available": self.train_available,
                         "val_probe": len(self.probe), **{s: len(v) for s, v in self.test.items()}},
            "train_directions": {DIR_LABEL[k]: v for k, v in Counter((e["src"], e["tgt"]) for e in self.train).items()},
            "train_instruction_languages": dict(Counter(e["lang"] for e in self.train)),
        }


def build_datasets(poems: list[dict], cfg: RunConfig) -> Datasets:
    P = cfg.preset
    by_split = assign_splits(poems, cfg)
    rng = random.Random(cfg.seed)
    langs, weights = list(cfg.lang_weights), list(cfg.lang_weights.values())
    pool = []
    for p in by_split["train"]:
        for i in range(len(shers(p, scripts_of(p)[0]))):
            for s, t in DIRECTIONS:
                lang = rng.choices(langs, weights=weights)[0]
                if (e := make_example(p, s, t, i, lang)):
                    pool.append(e)
    rng.shuffle(pool)
    train = pool[: P["n_train"]] if P["n_train"] else pool
    test = {s: eval_set(limit_poems(by_split[s], P["eval_poems"], cfg.seed)) for s in TEST_SPLITS}
    probe = [e for p in limit_poems(by_split["val"], P["probe_poems"], cfg.seed)
             for (s, t) in DIRECTIONS if (e := make_example(p, s, t, 0))]
    train_poems = {e["poem_id"] for e in train}
    assert not train_poems & {e["poem_id"] for v in test.values() for e in v}, "train/test overlap"
    return Datasets(by_split, train, len(pool), test, probe)
