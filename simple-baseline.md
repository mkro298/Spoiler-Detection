# Simple Baseline: Majority-Class Classifier

## Approach

Our simple baseline predicts the most common spoiler label in the training data for every test review. We count the number of reviews labeled `is_spoiler: true` and `is_spoiler: false` in `Reviews/train/IMDB_reviews.json`. If at least half of the training reviews are spoilers, we predict spoiler (`1`) for all test reviews; otherwise, we predict non-spoiler (`0`). Ties are resolved in favor of spoiler.

For ROC-AUC, we give every review the same spoiler score: the proportion of spoiler reviews in the training set. This does not use the review text or movie ID. Because all scores are equal, the baseline cannot rank spoilers above non-spoilers, so its ROC-AUC is 0.5 when both classes appear in the test data.

## Running the baseline

From the project root (with `score.py` in the same folder), run:

```bash
python3 simple-baseline.py --train Reviews/train/IMDB_reviews.json --test Reviews/test/IMDB_reviews.json
```

The script reads the files as JSON Lines (one review per line), prints training counts and evaluation metrics, and writes `simple-baseline-predictions.csv` with `predicted_label` and `spoiler_score` columns. Rows are in the original test-file order.

## Sample output

Output from running the baseline on the IMDb training and test sets:

```text
Training examples: 459130
Spoilers in training: 126007 (0.2744)
Majority prediction: 0
Test examples: 57392
Predictions saved to: simple-baseline-predictions.csv
precision: 0.0000
recall: 0.0000
f1: 0.0000
roc_auc: 0.5000
```

## Test-set results

The training set contains 459,130 reviews, of which 126,007 (27.44%) are labeled spoilers. Therefore, the majority class is non-spoiler (`0`), and the baseline predicts non-spoiler for all 57,392 test reviews. The measured test-set precision, recall, and F1 for the spoiler class are all **0.0000**, while **ROC-AUC is 0.5000**. This is expected for a constant-prediction baseline: it never identifies a spoiler, and its identical scores cannot rank spoiler reviews above non-spoiler reviews. These results provide a minimal reference point for evaluating stronger classifiers.
