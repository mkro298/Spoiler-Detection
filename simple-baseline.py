import argparse
import csv
import json

from score import evaluate

def read_reviews(path):
    # IMDb files have one JSON object per line (JSONL).
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Run a majority-class spoiler baseline")
    parser.add_argument("--train", default="Reviews/train/IMDB_reviews.json")
    parser.add_argument("--test", default="Reviews/test/IMDB_reviews.json")
    parser.add_argument("--output", default="simple-baseline-predictions.csv")
    args = parser.parse_args()

    spoiler_count = 0
    total = 0
    for review in read_reviews(args.train):
        spoiler_count += int(review["is_spoiler"])
        total += 1

    if total == 0:
        raise ValueError("Training file is empty")

    spoiler_rate = spoiler_count / total
    majority_label = int(spoiler_rate >= 0.5)  # ties predict spoiler
    print(f"Training examples: {total}")
    print(f"Spoilers in training: {spoiler_count} ({spoiler_rate:.4f})")
    print(f"Majority prediction: {majority_label}")

    gold_labels = []
    scores = []
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["predicted_label", "spoiler_score"])
        for review in read_reviews(args.test):
            writer.writerow([majority_label, spoiler_rate])
            if "is_spoiler" in review:
                gold_labels.append(int(review["is_spoiler"]))
            scores.append(spoiler_rate)

    print(f"Test examples: {len(scores)}")
    print(f"Predictions saved to: {args.output}")
    if len(gold_labels) == len(scores) and scores:
        if len(set(gold_labels)) == 2:
            metrics = evaluate(scores, gold_labels)
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")
        else:
            print("Cannot compute ROC-AUC: test labels contain only one class.")
    else:
        print("Test labels unavailable; skipping evaluation.")


if __name__ == "__main__":
    main()