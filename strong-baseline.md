# Strong Baseline: BiLSTM Review Classifier

## Description

Our strong baseline is a bidirectional LSTM (BiLSTM) binary classifier over
review text, predicting whether an IMDB review (`review_text`) is a spoiler
(`is_spoiler`). This follows the LSTM-based approach used by SpoilerNet
(Wan, Misra, Nakashole & McAuley, 2019), the closest related work identified
in our literature review, simplified to a single-sentence/document encoder
(no sentence-level attention or book-specific word weighting) so that it
remains a reasonable starting point rather than a paper-scale system.

**Architecture:** tokens are lowercased and split on `[a-z']+`, mapped
through a trainable embedding layer (dim 100), fed through a single-layer
bidirectional LSTM (hidden size 64 per direction), and the concatenated
final forward/backward hidden states are passed through a dropout layer
and a linear classifier producing a spoiler logit. The model is trained
with `BCEWithLogitsLoss`, with `pos_weight` set to the train-set negative/
positive class ratio to counteract class imbalance (only ~17-27% of
reviews are spoilers across splits).

This is a substantially stronger baseline than a majority-class or
keyword-matching baseline: it learns distributed representations of review
language and uses sequence context (ordering of words), rather than
treating the review as a bag of words or always predicting the majority
class.

## Data and a note on scale

The full `Reviews/train/IMDB_reviews.json` is ~776MB (1.38M reviews) per
the Wan et al. corpus. Local disk space on this machine could not
accommodate downloading that split via git-lfs alongside the rest of the
repo. As a result, **the model is trained on `Reviews/dev/IMDB_reviews.json`
(57,391 reviews)** instead of the official train split, and evaluated on
the official `Reviews/test/IMDB_reviews.json` (57,392 reviews) — the same
test file used for the simple baseline, so scores are directly comparable.
This is a resource-driven methodological choice, not a task-driven one;
training on the full 1.38M-review train split would very likely improve
these numbers further.

## How to run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python strong_baseline.py \
  --train Reviews/dev/IMDB_reviews.json \
  --test Reviews/test/IMDB_reviews.json \
  --epochs 3 \
  --predictions-out strong_baseline_predictions.jsonl
```

Key arguments (all optional, defaults shown):
`--vocab-size 20000`, `--max-len 100`, `--embed-dim 100`, `--hidden-dim 64`,
`--batch-size 128`, `--epochs 3`, `--lr 1e-3`, `--num-threads 10`.

The script loads both JSONL files, builds a vocabulary from the training
split, trains the BiLSTM for the given number of epochs, then runs
inference on the test split and calls `score.evaluate()` (from `score.py`)
directly to report precision, recall, F1, and ROC-AUC against the gold
`is_spoiler` labels. It also writes one JSON object per test example
(`{"score": ..., "is_spoiler": ...}`) to `--predictions-out`, which can be
re-scored independently with `score.py` as:

```python
import json, score
rows = [json.loads(l) for l in open("strong_baseline_predictions.jsonl")]
print(score.evaluate([r["score"] for r in rows], [r["is_spoiler"] for r in rows]))
```

## Sample output

```
57391 train examples, 57392 test examples
Train spoiler rate: 0.268
Test spoiler rate: 0.166
Vocab size: 20000
epoch 1/3 - train loss 0.9949
epoch 2/3 - train loss 0.9393
epoch 3/3 - train loss 0.8823
Test metrics: {
  "precision": 0.2858,
  "recall": 0.4800,
  "f1": 0.3583,
  "roc_auc": 0.6803
}
```

## Score on the test set

| Metric    | Value  |
|-----------|--------|
| Precision | 0.286  |
| Recall    | 0.480  |
| F1        | 0.358  |
| ROC-AUC   | 0.680  |

ROC-AUC of 0.68 shows the model ranks spoilers above non-spoilers
meaningfully better than chance (0.5), and well above the majority-class
simple baseline (which is undefined/trivial for ranking since it assigns
every example the same score). F1 of 0.358, with recall (0.48) notably
higher than precision (0.29), reflects the `pos_weight` term pushing the
model to flag more reviews as spoilers than the test set's 16.6% spoiler
rate would suggest by chance — a deliberate choice, since under-flagging
spoilers is the worse failure mode for this task's intended use case.
