"""Classify tweets, extract locations from disaster tweets, and plot them.

Usage:
    python predict_and_map.py                         # the Table II example tweets
    python predict_and_map.py --text "Wildfire spreading near Los Angeles"
    python predict_and_map.py --csv data/test.csv --limit 200

Output: outputs/disaster_map.html (open in a browser) and
        outputs/predictions.csv
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import tensorflow as tf

import config
from geolocate import extract_locations, plot_tweets
from preprocess import clean_text
from tokenizer import WordTokenizer

# Example tweets from Table II of the paper
TABLE_II_TWEETS = [
    "Snowstorm in Denver! Whiteout conditions, highways closed, and locals hunkering down",
    "Tornado warning in Oklahoma City! Sirens blaring, skies darkening, and residents seeking shelter.",
    "Flooding devastates New Orleans, Louisiana, as levees breach and neighborhoods are submerged. "
    "Residents evacuated as waters rise",
    "Severe flooding in Malawi displaces thousands, devastates crops. Relief efforts are underway. "
    "#MalawiFloods #DisasterRelief",
    "Just had the best pizza of my life, this place is fire!",   # non-disaster control
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.OUTPUT_DIR / "bilstm_glove.keras"))
    ap.add_argument("--text", nargs="*", help="one or more tweets to classify")
    ap.add_argument("--csv", help="CSV with a 'text' column (e.g. data/test.csv)")
    ap.add_argument("--limit", type=int, default=None, help="max rows to read from --csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=str(config.OUTPUT_DIR / "disaster_map.html"))
    args = ap.parse_args()

    if args.text:
        tweets = args.text
    elif args.csv:
        tweets = pd.read_csv(args.csv, nrows=args.limit)["text"].astype(str).tolist()
    else:
        tweets = TABLE_II_TWEETS

    model = tf.keras.models.load_model(args.model)
    tok = WordTokenizer.load(config.OUTPUT_DIR / "tokenizer.json")
    X = tok.encode([clean_text(t) for t in tweets], config.MAX_LEN)
    probs = model.predict(X, verbose=0).ravel()

    rows, disaster_records = [], []
    for text, p in zip(tweets, probs):
        is_disaster = p >= args.threshold
        # Location extraction only runs for tweets classified as disasters (Sec. III-E)
        locs = extract_locations(text) if is_disaster else []
        rows.append({"text": text, "probability": float(p),
                     "is_disaster": bool(is_disaster), "locations": "; ".join(locs)})
        label = "DISASTER" if is_disaster else "normal  "
        print(f"[{label} {p:.2f}] {text[:80]}")
        if locs:
            print(f"            locations: {locs}")
            disaster_records.append({"text": text, "probability": float(p), "locations": locs})

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / "predictions.csv", index=False)

    print(f"\nGeocoding {sum(len(r['locations']) for r in disaster_records)} location(s)...")
    plot_tweets(disaster_records, args.out)
    print(f"Map saved -> {args.out}")


if __name__ == "__main__":
    main()
