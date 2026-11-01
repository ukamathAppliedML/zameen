"""Importer on a tiny synthetic notebook shaped like the Colab notebook's outputs."""

import json

from zameen.importer import import_notebook


def _table(rows):
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return {"output_type": "display_data", "data": {"text/html": f"<table>{body}</table>"}}


def _cell(src, *outs):
    return {"cell_type": "code", "source": src, "outputs": list(outs)}


def test_import(tmp_path):
    base_row = ["Nastaliq → Devanagari", "30.0%", "28.0%", "50.0%", "0.0%", "0.0%", "10.0%", "80"]
    all_row = ["all directions", "25.0%", "20.0%", "50.0%", "0.5%", "0.0%", "4.0%", "480"]
    nb = {"cells": [
        _cell("MODEL = 'x'", {"output_type": "stream", "text": "Model Qwen/X · size small: {'n_train': 3000, 'eval_poems': 40, 'epochs': 1, 'probe_poems': 10}\n"}),
        _cell('PROG.start("baseline eval")', _table([["", "CER"], base_row, all_row]), _table([base_row, all_row])),
        _cell("trainer", {"output_type": "stream", "text": "3000 training sequences\ntrainable params: 68,812,800 || all\nUsing the best validation checkpoint: step 188 (val CER 2.9%)\n"}),
        _cell("# Learning curve", _table([["", "val_CER", "val_exact"], ["0", "27.0%", "0.0%"], ["3008", "2.9%", "38.0%"]])),
        _cell("def before_after(split)", *[_table([["Nastaliq → Devanagari", "30.0% → 5.0%", "28.0% → 4.0%", "50.0% → 15.0%", "0.0% → 12.0%", "-25%"],
                                                   ["all directions", "25.0% → 3.0%", "20.0% → 2.0%", "50.0% → 9.0%", "0.5% → 30.0%", "-22%"]])] * 2),
        _cell("for split in TEST:", *[_table([["Devanagari", "nukta", "10.0%", "80.0%", "+70%"], ["aspirates", "40.0%", "90.0%", "+50%"]])] * 2),
        _cell("rows = []", _table([["test_seen", "radif preserved", "50.0%", "95.0%"], ["qafiya preserved", "30.0%", "70.0%"],
                                   ["test_unseen", "radif preserved", "55.0%", "97.0%"], ["qafiya preserved", "31.0%", "80.0%"]])),
        _cell("summary = {", _table([["0", "training", "1017.5", "17.0"]])),
    ]}
    p = tmp_path / "run.ipynb"
    p.write_text(json.dumps(nb))
    s = import_notebook(p, tmp_path / "out", "x-small")
    d = s["splits"]["test_unseen"]["directions"]
    assert d["all directions"]["before"]["CER"] == 0.25 and d["all directions"]["after"]["exact"] == 0.30
    assert s["splits"]["test_seen"]["phenomena"]["Devanagari"]["aspirates"] == {"before": 0.4, "after": 0.9}
    assert s["splits"]["test_unseen"]["form"]["qafiya_kept"]["after"] == 0.8
    assert s["training"] == {"train_examples": 3000, "trainable_params": 68812800, "best_val_step": 188}
    assert s["validation_curve"][-1]["val_CER"] == 0.029 and s["timing_minutes"]["training"] == 17.0
    assert (tmp_path / "out" / "results.md").exists()
