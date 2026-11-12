Held-out test set (20% of train.csv, n=1523). Mean ± sd over seeds.

| Model          | Accuracy (%)   |   Precision |   Recall |   F1 Score |   Seeds |
|:---------------|:---------------|------------:|---------:|-----------:|--------:|
| BiLSTM + GloVe | 81.62 ± 0.81   |      0.8158 |   0.7413 |     0.7757 |       5 |
| BiGRU + GloVe  | 81.90 ± 0.99   |      0.8277 |   0.7327 |     0.7767 |       5 |
| GRU + GloVe    | 80.83 ± 1.58   |      0.8143 |   0.726  |     0.7641 |       5 |
| LSTM + GloVe   | 80.38 ± 1.39   |      0.8174 |   0.7067 |     0.7551 |       5 |
