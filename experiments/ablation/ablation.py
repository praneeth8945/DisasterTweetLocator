"""Ablation study for GloVe(6B.50d) + BiLSTM (original GLOVE_LSTM notebook configuration).

Split (fixed):
  test  = 20% of train.csv, random_state=42 (same split as the original notebook)
          -> never used for selection; scored only for reporting
  inner = the other 80%, split again 87.5/12.5 -> train / val
          -> val drives early stopping and config selection

Usage (from the repo root):
    python experiments/ablation/ablation.py '{"units": 64}' 42
    python experiments/ablation/ablation.py '{}' 1          # baseline config, seed 1
Appends one JSON line to ablation_results.jsonl. The 24 runs reported in the
paper are already in that file; see run_all.sh for the exact list.
"""
import json, os, re, string, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, optimizers, callbacks

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "train.csv"
GLOVE = ROOT / "data" / "glove.6B.50d.txt"
OUT = Path(__file__).resolve().parent / "ablation_results.jsonl"

DEFAULT = dict(text="raw", units=128, rec_dropout=0.2, maxlen=50, max_words=10000,
               dense=64, dropout=0.5, spatial=0.0, batch=64, lr=1e-3, epochs=15,
               patience=3, finetune_epochs=0, finetune_lr=1e-4, dedupe=False)

_url = re.compile(r"https?://\S+|www\.\S+")
def clean(t):
    t = t.lower(); t = re.sub(r"<.*?>", " ", t); t = _url.sub(" ", t)
    t = re.sub(r"\[.*?\]", " ", t); t = re.sub(r"[%s]" % re.escape(string.punctuation), " ", t)
    t = re.sub(r"\w*\d\w*", " ", t); return re.sub(r"\s+", " ", t).strip()

def make_text(df, mode):
    kw = df["keyword"].fillna("").str.replace("%20", " ")
    if mode == "raw": return df["text"]
    if mode == "clean": return df["text"].map(clean)
    if mode == "raw_kw": return kw + " " + df["text"]
    if mode == "clean_kw": return (kw + " " + df["text"]).map(clean)
    if mode == "clean_kw_url": # keep a marker token where a URL was
        return (kw + " " + df["text"].str.replace(_url, " httpurl ", regex=True)).map(clean)
    raise ValueError(mode)

_glove = None
def glove():
    global _glove
    if _glove is None:
        _glove = {}
        with open(GLOVE, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip().split(" ")
                if len(p) == 51: _glove[p[0]] = np.asarray(p[1:], dtype="float32")
    return _glove

def run(cfg, seed):
    c = {**DEFAULT, **cfg}
    tf.keras.utils.set_random_seed(seed)
    df = pd.read_csv(DATA)
    df["x"] = make_text(df, c["text"])
    rest, test = train_test_split(df, test_size=0.2, random_state=42)
    tr, val = train_test_split(rest, test_size=0.125, random_state=seed, stratify=rest["target"])
    if c["dedupe"]:  # training rows only: drop exact-duplicate texts with conflicting labels
        g = tr.groupby("text")["target"].transform("nunique")
        tr = tr[g == 1].drop_duplicates("text")
    tok = Tokenizer(num_words=c["max_words"]); tok.fit_on_texts(tr["x"].tolist())
    enc = lambda s: pad_sequences(tok.texts_to_sequences(s.tolist()), maxlen=c["maxlen"])
    Xtr, Xva, Xte = enc(tr["x"]), enc(val["x"]), enc(test["x"])
    ytr, yva, yte = tr["target"].to_numpy("int32"), val["target"].to_numpy("int32"), test["target"].to_numpy("int32")
    n = min(c["max_words"], len(tok.word_index)) + 1
    E = np.zeros((n, 50), dtype="float32"); G = glove()
    for w, i in tok.word_index.items():
        if i < n and w in G: E[i] = G[w]
    emb = layers.Embedding(n, 50, embeddings_initializer=tf.keras.initializers.Constant(E), trainable=False)
    L = [layers.Input((c["maxlen"],)), emb]
    if c["spatial"]: L.append(layers.SpatialDropout1D(c["spatial"]))
    L += [layers.Bidirectional(layers.LSTM(c["units"], dropout=0.2 if c["rec_dropout"] else 0.0,
                                           recurrent_dropout=c["rec_dropout"], return_sequences=True)),
          layers.GlobalMaxPooling1D(), layers.Dense(c["dense"], activation="relu"),
          layers.Dropout(c["dropout"]), layers.Dense(1, activation="sigmoid")]
    m = models.Sequential(L)
    m.compile(optimizer=optimizers.Adam(c["lr"]), loss="binary_crossentropy", metrics=["accuracy"])
    es = lambda: [callbacks.EarlyStopping(monitor="val_loss", patience=c["patience"], restore_best_weights=True)]
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=c["epochs"], batch_size=c["batch"], verbose=0, callbacks=es())
    ep = len(h.history["loss"])
    if c["finetune_epochs"]:
        emb.trainable = True
        m.compile(optimizer=optimizers.Adam(c["finetune_lr"]), loss="binary_crossentropy", metrics=["accuracy"])
        m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=c["finetune_epochs"], batch_size=c["batch"], verbose=0, callbacks=es())
    def score(X, y):
        p = (m.predict(X, verbose=0).ravel() >= 0.5).astype(int)
        return dict(acc=accuracy_score(y, p), prec=precision_score(y, p), rec=recall_score(y, p), f1=f1_score(y, p))
    return dict(cfg=cfg, seed=seed, epochs=ep, val=score(Xva, yva), test=score(Xte, yte))

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    t = time.time(); r = run(cfg, seed); r["sec"] = round(time.time() - t)
    with open(OUT, "a") as f: f.write(json.dumps(r) + "\n")
    print(f"val {r['val']['acc']:.4f} test {r['test']['acc']:.4f} f1 {r['test']['f1']:.4f} ep {r['epochs']} {r['sec']}s  {cfg}")
