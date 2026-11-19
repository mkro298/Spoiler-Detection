def to_label(value, where):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "spoiler"}:
        return 1
    if text in {"0", "false", "f", "no", "n", "not_spoiler", "non_spoiler",
                "nonspoiler", "not spoiler", "non-spoiler"}:
        return 0
    raise ValueError(f"Unrecognised binary label {value!r} ({where})")


def confusion(gold, pred):
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 0)
    return tp, fp, fn, tn


def safe_div(a, b):
    return a / b if b else 0.0


def precision(tp, fp):
    return safe_div(tp, tp + fp)


def recall(tp, fn):
    return safe_div(tp, tp + fn)


def roc_auc(gold, scores):
    n_pos = sum(gold)
    n_neg = len(gold) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC is undefined when gold contains only one class")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for r, g in zip(ranks, gold) if g == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate(system_outputs, gold_answers, threshold=0.5):
    """Return F1 and ROC-AUC for binary spoiler classification.

    ``system_outputs`` may be hard 0/1 labels or continuous spoiler scores
    (for example, probabilities). Scores are thresholded for F1 and retained
    as continuous values for ROC-AUC. Both arguments must be in the same order.
    """
    if len(system_outputs) != len(gold_answers):
        raise ValueError(
            "system_outputs and gold_answers must have the same number of items"
        )
    
    gold = [to_label(value, f"gold answer at index {i}")
            for i, value in enumerate(gold_answers)]
    scores = []
    for i, value in enumerate(system_outputs):
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            scores.append(float(to_label(value, f"system output at index {i}")))
    if any(not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("system outputs must be labels or scores between 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    predictions = [int(score >= threshold) for score in scores]
    tp, fp, fn, _tn = confusion(gold, predictions)
    precision_value = precision(tp, fp)
    recall_value = recall(tp, fn)
    f1 = safe_div(
        2 * precision_value * recall_value,
        precision_value + recall_value,
    )
    auc = roc_auc(gold, scores)
    return {
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1,
        "roc_auc": auc,
    }


def main():
    return


if __name__ == "__main__":
    main()
