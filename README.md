# Neural Networks for Binary Classification on Multimodal Data

An extensible PyTorch framework to experiment with neural-networks-based deep learning algorithms on multiple data
modalities for binary classification.

## Main features

* **Dynamic multi-modal architectures**. Off-the-shelf and easy-to-combine basic architectures (e.g. CNN, GRU, etc…) for
  multimodal approaches to binary classification tasks;
* **Structured experiments**. Run multiple reproducible iterations of cross-validation using different random seeds or
  metadata and generate detailed reports (including metrics, plots and predictions);
* **Data management**. Ready-to-use grouped data splits for K-fold, generated anew or based on metadata;
* **Ease of extension**. Clear extension points and easy customisation for different use cases.

## Structure of the project

Main high-level packages:

* `classifier`. The core of the project, including the fundamental classes handling the dataset, the training and
  evaluation of the models as well as the neural networks architectures;
* `preprocessing`. A pipeline-structured utility that allows for the generation and preprocessing of the data. For
  instance, image representations of eye-tracking data can be produced from the corresponding sequences, while the
  latter need to be adjusted and polished before being fed to a model;
* `scripts`. Various utility scripts for post-processing the results (e.g. aggregating the scores produced by multiple
  iterations of CV).
