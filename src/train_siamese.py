"""Train the ResNet-101 triplet model used for crack verification."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "triplet_crack_dataset"
IMAGE_SIZE = (227, 227)
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class TrainingConfig:
    image_size: Tuple[int, int] = IMAGE_SIZE
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 1e-4
    epsilon: float = 1e-1
    margin: float = 1.0
    seed: int = 42


def list_images(directory: Path) -> List[Path]:
    """Return image files in a stable order."""

    if not directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    images = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in {directory}")
    return images


def make_triplets(
    anchor_directory: Path,
    positive_directory: Path,
    seed: int,
    limit: int | None = None,
) -> List[Tuple[Path, Path, Path]]:
    """Pair anchors and positives, then sample deterministic negatives."""

    anchors = list_images(anchor_directory)
    positives = list_images(positive_directory)
    if len(anchors) != len(positives):
        raise ValueError(
            f"Anchor and positive counts differ: {len(anchors)} vs {len(positives)}"
        )
    if [path.name for path in anchors] != [path.name for path in positives]:
        raise ValueError("Anchor and positive filenames must match after sorting")
    if limit is not None:
        if limit < 3:
            raise ValueError("--limit must be at least 3")
        anchors = anchors[:limit]
        positives = positives[:limit]

    negative_pool = np.asarray(anchors + positives, dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(negative_pool)

    triplets: List[Tuple[Path, Path, Path]] = []
    for index, (anchor, positive) in enumerate(zip(anchors, positives)):
        candidate_index = index % len(negative_pool)
        negative = negative_pool[candidate_index]
        while negative in (anchor, positive):
            candidate_index = (candidate_index + 1) % len(negative_pool)
            negative = negative_pool[candidate_index]
        triplets.append((anchor, positive, Path(negative)))
    return triplets


def split_triplets(
    triplets: Sequence[Tuple[Path, Path, Path]], seed: int
) -> Dict[str, List[Tuple[Path, Path, Path]]]:
    """Create the 60/15/25 split reported in the paper."""

    shuffled = list(triplets)
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * TRAIN_FRACTION)
    validation_end = round(len(shuffled) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def decode_image(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, image_size) * 255.0


def preprocess_triplet(
    anchor: tf.Tensor,
    positive: tf.Tensor,
    negative: tf.Tensor,
    image_size: Tuple[int, int],
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return (
        decode_image(anchor, image_size),
        decode_image(positive, image_size),
        decode_image(negative, image_size),
    )


def make_dataset(
    triplets: Sequence[Tuple[Path, Path, Path]],
    config: TrainingConfig,
    shuffle: bool,
) -> tf.data.Dataset:
    if not triplets:
        raise ValueError("Cannot build a dataset from zero triplets")
    anchors, positives, negatives = zip(*triplets)
    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(path) for path in anchors], [str(path) for path in positives], [str(path) for path in negatives])
    )
    if shuffle:
        dataset = dataset.shuffle(len(triplets), seed=config.seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda a, p, n: (preprocess_triplet(a, p, n, config.image_size),),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


def build_embedding_model(
    image_size: Tuple[int, int], weights: str = "imagenet"
) -> tf.keras.Model:
    backbone_weights = None if weights == "none" else weights
    backbone = tf.keras.applications.ResNet101(
        weights=backbone_weights,
        input_shape=image_size + (3,),
        include_top=False,
    )

    features = tf.keras.layers.Flatten()(backbone.output)
    features = tf.keras.layers.Dense(256, activation="relu")(features)
    features = tf.keras.layers.BatchNormalization()(features)
    features = tf.keras.layers.Dense(128, activation="relu")(features)
    features = tf.keras.layers.BatchNormalization()(features)
    embedding = tf.keras.layers.Dense(128, name="embedding")(features)
    model = tf.keras.Model(backbone.input, embedding, name="embedding")

    trainable = False
    for layer in backbone.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable
    return model


class DistanceLayer(tf.keras.layers.Layer):
    """Compute squared distances from the anchor to positive and negative."""

    def call(self, anchor: tf.Tensor, positive: tf.Tensor, negative: tf.Tensor):
        positive_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        negative_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        return positive_distance, negative_distance


@tf.keras.utils.register_keras_serializable(package="crack_verification")
class SiameseModel(tf.keras.Model):
    def __init__(self, embedding_model: tf.keras.Model, margin: float, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        self.distance_layer = DistanceLayer()
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    @staticmethod
    def _inputs(data):
        # The dataset yields one tuple so Keras does not interpret the
        # negative image as a sample-weight tensor.
        return data[0] if isinstance(data, (tuple, list)) and len(data) == 1 else data

    def call(self, inputs, training=False):
        anchor, positive, negative = inputs
        anchor_embedding = self.embedding_model(
            tf.keras.applications.resnet.preprocess_input(anchor), training=training
        )
        positive_embedding = self.embedding_model(
            tf.keras.applications.resnet.preprocess_input(positive), training=training
        )
        negative_embedding = self.embedding_model(
            tf.keras.applications.resnet.preprocess_input(negative), training=training
        )
        return self.distance_layer(anchor_embedding, positive_embedding, negative_embedding)

    def compute_loss(self, data, training=False):
        positive_distance, negative_distance = self(self._inputs(data), training=training)
        loss = tf.maximum(positive_distance - negative_distance + self.margin, 0.0)
        return tf.reduce_mean(loss)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data, training=True)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.compute_loss(data, training=False)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def save_history(history: tf.keras.callbacks.History, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    history_path = output_directory / "history.csv"
    keys = sorted(history.history)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for index, values in enumerate(zip(*(history.history[key] for key in keys)), start=1):
            writer.writerow([index, *values])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "training_run")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-1)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Use only the first N pairs for a smoke run")
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("epochs and batch-size must be positive")

    tf.keras.utils.set_random_seed(args.seed)
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        margin=args.margin,
        seed=args.seed,
    )
    triplets = make_triplets(
        args.data_root / "anchor",
        args.data_root / "positive",
        seed=args.seed,
        limit=args.limit,
    )
    splits = split_triplets(triplets, seed=args.seed)
    print({name: len(items) for name, items in splits.items()})

    train_dataset = make_dataset(splits["train"], config, shuffle=True)
    validation_dataset = make_dataset(splits["validation"], config, shuffle=False)
    embedding_model = build_embedding_model(config.image_size, weights=args.weights)
    siamese_model = SiameseModel(embedding_model, margin=config.margin)
    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate, epsilon=config.epsilon
        )
    )
    history = siamese_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.epochs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_model.save(args.output_dir / "embedding.keras")
    siamese_model.save_weights(args.output_dir / "siamese_model.weights.h5")
    save_history(history, args.output_dir)
    print(f"Saved training outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
