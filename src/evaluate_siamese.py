"""Evaluate an embedding model on the held-out crack triplets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from train_siamese import (
    DEFAULT_DATA_ROOT,
    IMAGE_SIZE,
    TrainingConfig,
    make_dataset,
    make_triplets,
    split_triplets,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPOSITORY_ROOT / "artifacts" / "embedding6.keras"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "evaluation"


def manhattan_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs(first - second).sum(axis=1)


def calculate_distances(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    positive_distances: List[float] = []
    negative_distances: List[float] = []
    for batch in dataset:
        anchor, positive, negative = batch[0] if len(batch) == 1 else batch
        embeddings = [
            model(tf.keras.applications.resnet.preprocess_input(images), training=False).numpy()
            for images in (anchor, positive, negative)
        ]
        positive_distances.extend(manhattan_distance(embeddings[0], embeddings[1]))
        negative_distances.extend(manhattan_distance(embeddings[0], embeddings[2]))
    return np.asarray(positive_distances), np.asarray(negative_distances)


def evaluate_distances(
    positive_distances: np.ndarray,
    negative_distances: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    labels = np.concatenate(
        [np.ones(len(positive_distances), dtype=int), np.zeros(len(negative_distances), dtype=int)]
    )
    distances = np.concatenate([positive_distances, negative_distances])
    predictions = (distances <= threshold).astype(int)
    scores = -distances
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, scores)),
        "positive_mean": float(np.mean(positive_distances)),
        "positive_std": float(np.std(positive_distances)),
        "negative_mean": float(np.mean(negative_distances)),
        "negative_std": float(np.std(negative_distances)),
    }


def write_outputs(
    positive_distances: np.ndarray,
    negative_distances: np.ndarray,
    metrics: Dict[str, float],
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "distances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "positive_distance", "negative_distance"])
        for index, (positive, negative) in enumerate(zip(positive_distances, negative_distances)):
            writer.writerow([index, float(positive), float(negative)])
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=10.56)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N triplets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Embedding model not found: {args.model}. Train a model or pass --model."
        )
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")

    triplets = make_triplets(
        args.data_root / "anchor",
        args.data_root / "positive",
        seed=args.seed,
        limit=args.limit,
    )
    test_triplets = split_triplets(triplets, seed=args.seed)["test"]
    config = TrainingConfig(image_size=IMAGE_SIZE, batch_size=args.batch_size, seed=args.seed)
    dataset = make_dataset(test_triplets, config, shuffle=False)
    model = tf.keras.models.load_model(args.model, compile=False)
    positive, negative = calculate_distances(model, dataset)
    metrics = evaluate_distances(positive, negative, args.threshold)
    write_outputs(positive, negative, metrics, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
