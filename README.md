# Triplet Loss-Based Concrete Crack Verification

An embedding-based method for deciding whether two concrete-crack images represent the same crack pattern. The model uses a shared ResNet-101 encoder and triplet loss.

## Overview

Conventional crack detection asks whether a crack is present. This work addresses a different question: **is the crack observed now the same crack that was observed before?**

That distinction matters in Structural Health Monitoring (SHM). A detector can flag damage in one image, but it does not establish the identity of that damage across observations. The proposed system maps crack images into an embedding space and uses embedding distance for verification.

## Method at a Glance

```text
Anchor + Positive + Negative
             |
             v
     shared ResNet-101 encoder
             |
             v
       128-dimensional embeddings
             |
             v
        Triplet loss and distances
             |
             v
   similar patterns closer, different patterns farther apart
```

The training triplet contains an anchor image, a positive image with a similar crack pattern, and a negative image with a different pattern. The same encoder processes all three images. During inference, two images are encoded and compared using their embedding distance.

## Network Architecture

The model is a Siamese network with a shared ResNet-101 feature extractor. The paper describes a pre-trained ResNet-101 backbone followed by dense layers of 256 and 128 units, each with ReLU and batch normalization, and a final 128-dimensional embedding output.

The triplet loss is:

```text
L = max(0, d(a, p) - d(a, n) + margin)
```

It penalizes cases where the negative is not farther from the anchor than the positive by at least the chosen margin.

## Dataset and Training

The paper uses the 20,000-image Crack Dataset and prepares it as triplets. It reports 12,000 images for training, 3,000 for validation, and 5,000 for testing. Positive examples are generated through transformations of anchor images, including rotation and shearing. They are not real repeated observations of the same physical crack.

The reported configuration is:

| Setting | Value |
| --- | --- |
| Input size | 227 × 227 pixels |
| Batch size | 32 |
| Epochs | 200 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Epsilon | 1e-1 |
| Triplet-loss margin | 1 |
| Dense layers | 256, then 128 units; ReLU + batch normalization |
| Embedding size | 128 |

The local working copy is under [`data/triplet_crack_dataset`](data/triplet_crack_dataset). The dataset directory and the saved embedding model are kept locally and excluded from a normal GitHub checkout because of their size.

## Code

The main workflow is implemented as regular Python scripts under [`src/`](src/):

- `prepare_triplet_dataset.py` creates transformed positive images with deterministic random settings.
- `train_siamese.py` builds the shared ResNet-101 encoder, creates triplets, trains the model, and saves the embedding model and history.
- `evaluate_siamese.py` loads an embedding model, computes Manhattan distances, and writes distances and verification metrics.

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Create positive examples when starting from the anchor images:

```bash
python src/prepare_triplet_dataset.py \
  --anchor-dir data/triplet_crack_dataset/anchor \
  --positive-dir data/triplet_crack_dataset/positive
```

Train with the configuration reported in the paper:

```bash
python src/train_siamese.py
```

For a short local smoke run, use a small subset and disable ImageNet weight loading:

```bash
python src/train_siamese.py --limit 24 --epochs 1 --weights none \
  --output-dir outputs/smoke_test
```

Evaluate a saved embedding model:

```bash
python src/evaluate_siamese.py \
  --model outputs/training_run/embedding.keras
```

The original notebooks remain in [`research/notebooks_archive`](research/notebooks_archive) for provenance. They are not the primary execution path.

## Results

The paper reports the following image-verification metrics:

| Metric | Reported value |
| --- | ---: |
| Accuracy | 97.36% |
| Precision | 95.77% |
| Recall | 99.1% |
| F1 score | 97.41% |
| AUC | 99.61% |

The paper reports mean Manhattan distances of 5.13 (standard deviation 2.01) for positive pairs and 19.89 (standard deviation 5.75) for negative pairs. The reported optimal embedding-distance threshold is 10.56.

These results evaluate image verification on the prepared triplet data. They do not validate deployment in a live SHM system, real-time operation, UAV capture, or a complete Digital Twin implementation.

## Why This Matters for Structural Health Monitoring

Crack detection answers whether damage is visible in one image. Crack verification adds an identity question: whether a later observation corresponds to the same physical damage. If that identity can be established reliably, future monitoring workflows could compare the crack's appearance over time instead of treating every detection as unrelated damage.

The paper presents this as a direction for structural monitoring and digital-twin applications. The repository does not claim that longitudinal crack-evolution tracking was experimentally validated.

## Limitations

The paper identifies a central limitation: positive training images were generated by transforming anchor images rather than by collecting real repeated observations of the same crack. Real longitudinal observations would be an important next step for testing transfer to changing field conditions and generalization beyond synthetic positive pairs.

## Citation

```bibtex
@inproceedings{neyestani2024triplet,
  author    = {Neyestani, Arman and Picariello, Francesco and Tudosa, Ioan and Daponte, Pasquale and De Vito, Luca},
  title     = {Triplet Loss-Based Concrete Crack Verification for Structural Health Monitoring and Digital Twin Applications},
  booktitle = {2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)},
  year      = {2024},
  pages     = {837--842},
  doi       = {10.1109/MetroXRAINE62247.2024.10797100}
}
```

The source paper is available at [`paper/triplet_loss_crack_verification.pdf`](paper/triplet_loss_crack_verification.pdf).

## Authors and Acknowledgment

Arman Neyestani, Francesco Picariello, Ioan Tudosa, Pasquale Daponte, and Luca De Vito, Department of Engineering, University of Sannio.

The paper acknowledges partial support from the NATO Science for Peace and Security Programme Multi-Year Project G5924, “Inspection and security by Robots interacting with Infrastructure digital twins” (IRIS).

The repository code is released under the MIT License. The paper and third-party dataset content retain their original rights and terms; see [`LICENSE`](LICENSE) for the repository-code license.
