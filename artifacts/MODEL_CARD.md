# Released model artifact

`embedding6.keras` is the trained 128-dimensional embedding model used for
the repository's verification examples. It contains the ResNet-101 backbone
and the projection layers described in the paper. It is saved as a Keras model
and can be loaded with `tf.keras.models.load_model`.

The matching `embedding6.weights.h5` file contains the model weights in HDF5
format. The `.keras` file is the easiest option for inference; the HDF5 file
is useful when the architecture is recreated in code before loading weights.

## Intended use

Use the model to encode two 227 × 227 RGB images and compare their embedding
vectors. The paper reports a Manhattan-distance threshold of 10.56 for its
prepared evaluation setup. Recalibrate that threshold for a new camera,
surface, or dataset before using it in a monitoring workflow.

## Training provenance

- Backbone: ResNet-101 with the paper's dense projection layers.
- Output: 128-dimensional embedding.
- Training data: the prepared triplet data documented in `data/README.md`.
- Source data license: CC BY 4.0; retain the source attribution when using
  the model with the accompanying dataset.

This checkpoint is a research artifact. It does not establish longitudinal
tracking performance on real repeated observations of the same structure.
