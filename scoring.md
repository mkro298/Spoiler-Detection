# Evaluation Metrics

This project tackles spoiler detection as a binary classification task. The
positive class is **spoiler** (`1`); the negative class is **not spoiler** (`0`).

## Precision, recall, and F1

Precision is the fraction of predicted spoilers that are actually spoilers:

\[
P = \frac{TP}{TP + FP}.
\]

Recall is the fraction of gold spoilers the system correctly identifies:

\[
R = \frac{TP}{TP + FN}.
\]

F1 is the harmonic mean of precision and recall, weighting them equally:

\[
F_1 = 2 \cdot \frac{P \cdot R}{P + R}
    = \frac{2TP}{2TP + FP + FN}.
\]

F1 ranges from 0 to 1, with larger values indicating a better balance between
precision and recall. This is the standard F-measure with equal weighting of
precision and recall, described by van Rijsbergen.
[van Rijsbergen, *Information Retrieval* (1979)](https://books.google.com/books?id=t-pTAAAAMAAJ);
[F-score overview](https://en.wikipedia.org/wiki/F-score).

## ROC-AUC

The receiver operating characteristic (ROC) curve plots the true positive
rate (TPR) against the false positive rate (FPR) over all possible score
thresholds:

\[
TPR = \frac{TP}{TP + FN}, \qquad
FPR = \frac{FP}{FP + TN}.
\]

ROC-AUC is the area under this curve. It measures how well the system ranks a
random gold spoiler above a random gold non-spoiler: 1 is perfect ranking, 0.5
is chance-level ranking, and 0 is perfectly reversed ranking. Unlike F1,
ROC-AUC uses the system's continuous spoiler scores and does not depend on one
classification threshold.
[Hanley and McNeil (1982), “The meaning and use of the area under a receiver
operating characteristic (ROC) curve”](https://doi.org/10.1148/radiology.143.1.7063747).

## How `score.py` applies the metrics

Call `evaluate(system_outputs, gold_answers, threshold=0.5)`. The function
returns precision, recall, F1, and ROC-AUC.
