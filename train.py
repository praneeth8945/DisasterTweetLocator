"""Train and evaluate the four GloVe models from Table I.

Every model (BiLSTM, BiGRU, GRU, LSTM) is trained once per seed in
config.SEEDS and scored on a fixed 20% held-out test split that is never used
for training, early stopping or model selection. Table I reports
mean ± standard deviation over the seeds.

Usage:
    python train.py                         # full experiment (4 models x 5 seeds)
    python train.py --models bilstm --seeds 42
    python train.py --summarize-only        # rebuild the tables from outputs/runs.jsonl

Finished runs are appended to outputs/runs.jsonl and skipped on re-run, so an
interrupted experiment resumes where it stopped.

Outputs (outputs/):
    runs.jsonl                    one line per (model, seed) with val + test metrics
    results_table.md / .csv       Table I (mean ± sd over seeds, held-out test)
    results.json                  same numbers, machine-readable
    history_<model>.json          per-epoch loss/accuracy for the main seed
    loss_curves_<model>.png       training + validation loss (Fig. 4 & 5)
    bilstm_glove.keras            BiLSTM model from the main seed (used by predict_and_map.py)
    tokenizer.json                tokenizer for that model
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import config
from embeddings import build_embedding_matrix, load_glove
from models import DISPLAY_NAMES, RNN_VARIANTS, build_model
from preprocess import load_dataset
from tokenizer import WordTokenizer

RUNS_FILE = config.OUTPUT_DIR / "runs.jsonl"


def set_seed(seed: int):
    tf.keras.utils.set_random_seed(seed)


def split_data(df, seed):
    """Fixed held-out test split + per-seed train/validation split."""
    rest, test = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.TEST_SPLIT_SEED, stratify=df["target"]
    )
    train, val = train_test_split(
        rest, test_size=config.VAL_SIZE, random_state=seed, stratify=rest["target"]
    )
    return train, val, test


def evaluate(model, X, y):
    prob = model.predict(X, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion": confusion_matrix(y, pred).tolist(),
    }


def plot_losses(history: dict, variant: str):
    """Fig. 4 (validation loss) and Fig. 5 (training loss)."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["val_loss"], marker="o", color="#c0392b")
    axes[0].set_title(f"Validation Loss — {DISPLAY_NAMES[variant]}")
    axes[1].plot(epochs, history["loss"], marker="o", color="#2471a3")
    axes[1].set_title(f"Training Loss — {DISPLAY_NAMES[variant]}")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary cross-entropy")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / f"loss_curves_{variant}.png", dpi=150)
    plt.close(fig)


def done_runs():
    if not RUNS_FILE.exists():
        return set()
    return {(r["model"], r["seed"]) for r in map(json.loads, RUNS_FILE.read_text().splitlines())}


def run_seed(df, seed, variants, glove):
    train, val, test = split_data(df, seed)
    tok = WordTokenizer(config.MAX_VOCAB).fit(train["clean_text"])
    enc = lambda d: tok.encode(d["clean_text"].tolist(), config.MAX_LEN)
    X_tr, X_val, X_te = enc(train), enc(val), enc(test)
    y_tr, y_val, y_te = (d["target"].to_numpy("int32") for d in (train, val, test))
    emb, coverage = build_embedding_matrix(tok.word_index, glove, config.EMBEDDING_DIM, config.MAX_VOCAB)
    print(f"\n### seed {seed} | train {len(train)} val {len(val)} test {len(test)} | GloVe coverage {coverage:.1%}")
    if seed == config.MAIN_SEED:
        tok.save(config.OUTPUT_DIR / "tokenizer.json")

    for variant in variants:
        set_seed(seed)
        model = build_model(variant, emb)
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOP_PATIENCE, restore_best_weights=True
        )
        hist = model.fit(
            X_tr, y_tr, validation_data=(X_val, y_val), epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE, callbacks=[es], verbose=0,
        ).history
        hist = {k: [float(v) for v in vals] for k, vals in hist.items()}
        best_epoch = int(np.argmin(hist["val_loss"])) + 1
        rec = {
            "model": variant, "seed": seed, "best_epoch": best_epoch, "epochs_run": len(hist["loss"]),
            "val": evaluate(model, X_val, y_val), "test": evaluate(model, X_te, y_te),
        }
        with open(RUNS_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
        t = rec["test"]
        print(f"{DISPLAY_NAMES[variant]:16s} seed {seed}: test acc {t['accuracy']:.4f} prec {t['precision']:.4f} "
              f"rec {t['recall']:.4f} f1 {t['f1']:.4f} | best epoch {best_epoch}/{len(hist['loss'])}")
        if seed == config.MAIN_SEED:
            (config.OUTPUT_DIR / f"history_{variant}.json").write_text(json.dumps(hist))
            plot_losses(hist, variant)
            if variant == "bilstm":
                model.save(config.OUTPUT_DIR / "bilstm_glove.keras")


def summarize():
    runs = [json.loads(l) for l in RUNS_FILE.read_text().splitlines()]
    rows, summary = [], {}
    for variant in RNN_VARIANTS:
        rs = [r for r in runs if r["model"] == variant]
        if not rs:
            continue
        m = {k: np.array([r["test"][k] for r in rs]) for k in ("accuracy", "precision", "recall", "f1")}
        summary[variant] = {
            "n_seeds": len(rs), "seeds": sorted(r["seed"] for r in rs),
            **{f"{k}_mean": float(v.mean()) for k, v in m.items()},
            **{f"{k}_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0 for k, v in m.items()},
            "accuracy_min": float(m["accuracy"].min()), "accuracy_max": float(m["accuracy"].max()),
        }
        s = summary[variant]
        rows.append({
            "Model": DISPLAY_NAMES[variant],
            "Accuracy (%)": f"{s['accuracy_mean']*100:.2f} ± {s['accuracy_sd']*100:.2f}",
            "Precision": f"{s['precision_mean']:.4f}",
            "Recall": f"{s['recall_mean']:.4f}",
            "F1 Score": f"{s['f1_mean']:.4f}",
            "Seeds": len(rs),
        })
    table = pd.DataFrame(rows)
    table.to_csv(config.OUTPUT_DIR / "results_table.csv", index=False)
    (config.OUTPUT_DIR / "results_table.md").write_text(
        "Held-out test set (20% of train.csv, n=1523). Mean ± sd over seeds.\n\n" + table.to_markdown(index=False) + "\n"
    )
    (config.OUTPUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    print("\nTABLE I — MODEL EVALUATION METRICS (held-out test, mean over seeds)")
    print(table.to_string(index=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(RNN_VARIANTS), choices=RNN_VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    if not args.summarize_only:
        df = load_dataset(config.TRAIN_CSV)
        print(f"Loaded {len(df)} tweets | class balance {df['target'].value_counts().to_dict()}")
        glove = load_glove(config.GLOVE_PATH, config.EMBEDDING_DIM)
        finished = done_runs()
        for seed in args.seeds:
            todo = [v for v in args.models if (v, seed) not in finished]
            if todo:
                run_seed(df, seed, todo, glove)
    summarize()


if __name__ == "__main__":
    main()
