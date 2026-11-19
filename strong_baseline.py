"""BiLSTM strong baseline for review-level spoiler detection.

Trains a bidirectional LSTM binary classifier over IMDB review text
(review_text -> is_spoiler), in the spirit of the LSTM-based SpoilerNet
baseline from Wan et al. (2019). See strong-baseline.md for details.
"""
import argparse
import json
import re
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, ".")
import score as scoring

PAD, UNK = 0, 1
TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_reviews(path):
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["review_text"])
            labels.append(1 if row["is_spoiler"] else 0)
    return texts, labels


def build_vocab(token_lists, vocab_size):
    counts = Counter(tok for toks in token_lists for tok in toks)
    most_common = counts.most_common(vocab_size - 2)
    vocab = {tok: i + 2 for i, (tok, _) in enumerate(most_common)}
    return vocab


def encode(tokens, vocab, max_len):
    ids = [vocab.get(tok, UNK) for tok in tokens[:max_len]]
    return ids


class ReviewDataset(Dataset):
    def __init__(self, encoded, labels):
        self.encoded = encoded
        self.labels = labels

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx], self.labels[idx]


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = [max(len(s), 1) for s in seqs]
    max_len = max(lengths)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return padded, torch.tensor(lengths, dtype=torch.long), torch.tensor(labels, dtype=torch.float)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        final = torch.cat([hidden[0], hidden[1]], dim=1)
        return self.classifier(self.dropout(final)).squeeze(1)


def run_epoch(model, loader, optimizer, criterion, device, train):
    model.train(train)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_scores(model, loader, device):
    model.eval()
    scores = []
    for x, lengths, _ in loader:
        x, lengths = x.to(device), lengths.to(device)
        logits = model(x, lengths)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="Reviews/dev/IMDB_reviews.json",
                         help="JSONL file of reviews used to fit the model")
    parser.add_argument("--test", default="Reviews/test/IMDB_reviews.json",
                         help="JSONL file of reviews used for held-out evaluation")
    parser.add_argument("--max-train-examples", type=int, default=None,
                         help="optionally subsample the training file for speed")
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=10)
    parser.add_argument("--predictions-out", default=None,
                         help="optional path to write per-example scores as JSONL")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading train data from {args.train} ...")
    train_texts, train_labels = load_reviews(args.train)
    if args.max_train_examples:
        train_texts = train_texts[: args.max_train_examples]
        train_labels = train_labels[: args.max_train_examples]
    print(f"Loading test data from {args.test} ...")
    test_texts, test_labels = load_reviews(args.test)

    print(f"{len(train_texts)} train examples, {len(test_texts)} test examples")
    print(f"Train spoiler rate: {sum(train_labels) / len(train_labels):.3f}")
    print(f"Test spoiler rate: {sum(test_labels) / len(test_labels):.3f}")

    train_tokens = [tokenize(t) for t in train_texts]
    test_tokens = [tokenize(t) for t in test_texts]

    vocab = build_vocab(train_tokens, args.vocab_size)
    print(f"Vocab size: {len(vocab) + 2}")

    train_encoded = [encode(toks, vocab, args.max_len) for toks in train_tokens]
    test_encoded = [encode(toks, vocab, args.max_len) for toks in test_tokens]

    train_loader = DataLoader(
        ReviewDataset(train_encoded, train_labels),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    test_loader = DataLoader(
        ReviewDataset(test_encoded, test_labels),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate,
    )

    model = BiLSTMClassifier(len(vocab) + 2, args.embed_dim, args.hidden_dim).to(device)
    pos_weight = torch.tensor([(len(train_labels) - sum(train_labels)) / max(sum(train_labels), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        print(f"epoch {epoch}/{args.epochs} - train loss {train_loss:.4f}")

    scores = predict_scores(model, test_loader, device)
    metrics = scoring.evaluate(scores, test_labels)
    print("Test metrics:", json.dumps(metrics, indent=2))

    if args.predictions_out:
        with open(args.predictions_out, "w") as f:
            for score_value, label in zip(scores, test_labels):
                f.write(json.dumps({"score": score_value, "is_spoiler": label}) + "\n")
        print(f"Wrote per-example scores to {args.predictions_out}")


if __name__ == "__main__":
    main()
