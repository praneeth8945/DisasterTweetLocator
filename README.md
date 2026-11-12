# DisasterTweetLocator

**Disaster tweet classification with GloVe + BiLSTM, and geolocation mapping of the places those tweets mention.**

Twitter is one of the first places people report emergencies, but most tweets are noise and only 1–2% carry GPS coordinates. DisasterTweetLocator:

1. **Classifies** tweets as *disaster* or *normal* with a BiLSTM on pretrained GloVe embeddings
2. **Extracts** place names from disaster tweets with spaCy named-entity recognition
3. **Geocodes** them with geopy (Nominatim) and **plots** them on an interactive Folium map

📄 Paper: [`paper/main.pdf`](paper/main.pdf) · 📓 Walkthrough with all outputs: [`notebooks/DisasterTweetLocator.ipynb`](notebooks/DisasterTweetLocator.ipynb)

---

## Results

Held-out test set: 20% of the Kaggle training data (n = 1,523 tweets), never used for training, early stopping or model selection. Each model was trained with **5 random seeds**; the table shows the mean, with accuracy as mean ± standard deviation.

| Model | Accuracy (%) | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **BiLSTM + GloVe** | **81.62 ± 0.81** | 0.8158 | 0.7413 | 0.7757 |
| BiGRU + GloVe | 81.90 ± 0.99 | 0.8277 | 0.7327 | 0.7767 |
| GRU + GloVe | 80.83 ± 1.58 | 0.8143 | 0.7260 | 0.7641 |
| LSTM + GloVe | 80.38 ± 1.39 | 0.8174 | 0.7067 | 0.7551 |

- Bidirectional models beat unidirectional ones by about 1 point and vary less across seeds. BiLSTM and BiGRU are statistically tied.
- Precision is consistently higher than recall: the model rarely raises false alarms, but misses some disaster tweets that use indirect language. Seed-42 BiLSTM confusion matrix: TP 445, TN 799, FP 70, FN 209.
- **Ablation (24 runs).** Varying preprocessing, LSTM size (32–256), dropout, sequence length, vocabulary size, batch size, learning rate, label de-duplication and embedding fine-tuning gave **81.35–83.26%** held-out accuracy. The baseline config scored 82.18% ± 0.38 over 5 seeds. Performance is stable at about 82%. See [`experiments/ablation/`](experiments/ablation).

Per-seed numbers: [`outputs/runs.jsonl`](outputs/runs.jsonl). Loss curves: [`outputs/loss_curves_*.png`](outputs).

### Location extraction (paper Table II)

| Tweet | P(disaster) | Locations found |
|---|---|---|
| Snowstorm in Denver! Whiteout conditions, highways closed… | 0.86 | Denver |
| Tornado warning in Oklahoma City! Sirens blaring… | 1.00 | Oklahoma City |
| Flooding devastates New Orleans, Louisiana, as levees breach… | 1.00 | New Orleans, Louisiana |
| Severe flooding in Malawi displaces thousands… | 0.99 | *(not recognised by spaCy's small model)* |
| Just had the best pizza of my life, this place is fire! | 0.09 → normal | — |

The interactive map is saved to [`outputs/disaster_map.html`](outputs/disaster_map.html). Download it and open it in a browser; GitHub doesn't render it.

---

## Model

```
Input (50 tokens)
 → Embedding: GloVe 6B.50d, frozen, 10k vocab
 → Bidirectional LSTM (128 units per direction, return_sequences)
 → GlobalMaxPooling1D
 → Dense(64, ReLU)
 → Dropout(0.5)
 → Dense(1, sigmoid)
Adam (lr 0.001) · binary cross-entropy · batch 64 · early stopping on val_loss (patience 3)
```

## Repository layout

```
├── config.py              hyperparameters, paths, evaluation protocol
├── preprocess.py          loading and text cleaning (HTML, URLs, brackets, digits, punctuation)
├── tokenizer.py           word tokenizer (saved as JSON)
├── embeddings.py          GloVe loading and embedding matrix
├── models.py              BiLSTM / BiGRU / GRU / LSTM architectures
├── train.py               5-seed experiment, Table I, loss curves, saved model
├── geolocate.py           spaCy NER, geopy geocoding, Folium map
├── predict_and_map.py     classify tweets, then map the disaster ones
├── notebooks/             executed walkthrough notebook (all outputs visible)
├── experiments/ablation/  24-run ablation study (script + results)
├── outputs/               trained BiLSTM, tokenizer, results, curves, map
├── paper/                 LaTeX source and compiled PDF
└── data/                  Kaggle train.csv / test.csv, offline geocode cache
```

## Reproduce

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# GloVe 6B.50d (not included: 171 MB)
curl -L -o data/glove50.gz https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz
gunzip -c data/glove50.gz > data/glove.6B.50d.txt
#   (or download glove.6B.zip from https://nlp.stanford.edu/projects/glove/)

python train.py                  # 4 models x 5 seeds, ~20 min on CPU; resumes if interrupted
python predict_and_map.py        # Table II tweets -> outputs/disaster_map.html
python predict_and_map.py --text "Wildfire spreading near Los Angeles, homes evacuated"
python paper/make_results_tex.py # regenerate the paper's numbers, Table I and Fig. 4/5
```

`outputs/` already contains a trained model, so `predict_and_map.py` and the notebook run without retraining. Exact numbers can shift slightly across TensorFlow versions and hardware. Results above were produced with TensorFlow 2.21 on CPU.

## Dataset

[Kaggle: Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started): 7,613 labelled tweets (`id, keyword, location, text, target`) plus 3,263 unlabelled test tweets. The CSVs are included in `data/` for convenience; check Kaggle's competition rules before redistributing.

## Limitations and future work

- spaCy's `en_core_web_sm` misses some place names (e.g. *Malawi*); `en_core_web_trf` or a fine-tuned NER model should help.
- Recall is lower than precision. Class weighting or threshold tuning would trade some precision for recall.
- Transformer classifiers (BERT/RoBERTa) are a natural next step for higher accuracy.
