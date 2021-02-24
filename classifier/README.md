# Classifier

> This code is part of the project “Neural Networks for binary classification on multiple data modalities”. 

The `classifier` package is the core of the project, including the fundamental classes which handle the datasets and the train and evaluation of the models as well as the Neural Networks (NNs) architectures.

## Requirements

The code is based on PyTorch and can run both on CPU and GPU supporting CUDA. In order to run the software, please install the requirements listed in the `requirements.txt` file. The code has been tested using PyTorch 1.4.0.

## Framework Components

<img src="docs/gallery/framework-classifier.png" alt="class" style="zoom:75%;" />

### Core

The **core** component runs an experiment, which is a series of Cross-Validation (CV) procedures. It interacts with `data` and networks-related classes to fetch the data from the dataset and to train the model respectively. The main classes of the package are:

* `Trainer`: receives the `train_params` saved in the `experiment.json` file and handles the train of the model; 
* `Evaluator`: evaluates the model computing the reported metrics; 
* `Model`: takes care of prediction and weight update and is subclassed by each new module. 

### Modules

Modules consist of two parts:

* A *network* (subclassing torch `nn.Module`) defining the architecture of the NN and its forward phase (e.g. `GRU(nn.Module)`)
* A *model* (subclassing the base `Model` in the core component) defining how the input is fed to the network for the prediction (e.g. `ModelGRU(Model)`)

They can either handle:

* *Single modalities* (e.g. text as in BERT), or
* *Multiple modalities* (e.g. eye-tracking sequences and images as in the VisTempNet).

Multi modal networks must subclass the base `MultimodalNetwork` stored at `modules/base/networks`. Networks handling multiple modalities are built combining submodules (i.e. networks handling single modalities, e.g. VisTempNet = CNN + GRU) according to a features fusion policy (i.e. early fusion, late model blending, etc…) as exemplified by the following scheme. 

<img src="docs/gallery/multimodal-architecture.png" alt="image-20200715124444532" style="zoom:67%;" />

### Data

The data component handles:

* Data grouping for each dataset;
* Data loading for each modality;
* Data split for K-Fold CV.

## Dataset

In order to be used together with this project, a dataset must follow a precise structure:

1. The data items are grouped by **modality**. The currently supported modalities are:

   * *Sequences*: time-series;

   * *Images*: any image representation;

   * *Text*: word sequences properly encoded depending on the network which is processing them.

2. Each modality may have one or more **data sources** (e.g. the images may have two data sources: audio and eye-tracking;

3. Each data source may have one ore more **representations** (e.g. the eye-tracking images may be either scan-paths or heatmaps). 

The following scheme exemplifies the relationship among modalities, data sources and representations.

<img src="docs/gallery/data-organization.png" alt="image-20200729114506169" style="zoom:67%;" /> 

The dataset must also contain:

* A `metadata` folder where a `dataset.csv` file summarising the information about the data is dynamically generated;
* A `split` folder containing the metadata for the CV and where the data split is dynamically generated.

The metadata for generating the splits are CSV files stating which items belong to each set in each split and where each row correspond to a CV split. The metadata must be structured as follows:

```csv
train_pos , train_neg , test_pos  , test_neg
EO-083 ..., HE-099 ..., EA-084 ..., HP-138 ...
EO-091 ..., HA-113 ..., EE-190 ..., HH-055 ...
...
```

The general structure of a compatible dataset is the following:

```shell
├── metadata
├── modalities
│   ├── modality_1
│   │   └── data_source_1 [optional]
│   │   |   ├── representation_1 [optional]
│   │   |   │   └── base
│   │   |   │	|   ├── 0_neg_label
│   │   |   │   |   └── 1_pos_label
│   │   |   │   └── augmented
│   │   |   │       ├── 0_neg_label
│   │   |   │       └── 1_pos_label
|   |	|   |	
│   │   |   |	...
|   |	|   |	
|   |	|   └── representation_N [optional]
|   |   |
│   │   |   ...
|   |   |
|   |   └── data_source_N [optional]
|   | 	
|   | 	...
|   | 	
|   └── modality_N
└── split
    ├── folds [dynamically generated]
    │   ├── fold_1
    |	|
    |	|	...
    |	|
    │   └── fold_N
    └── metadata
```

For example:

```
├── metadata
├── modalities
│   ├── audio
│   ├── images
│   │   └── eye_tracking
│   │       ├── heatmaps
│   │       │   └── base
│   │       │       ├── 0_healthy
│   │       │       └── 1_alzheimer
│   │       └── scan_paths
│   │       	└── augmented
│   │               ├── 0_healthy
│   │               └── 1_alzheimer
│   ├── sequences
│   │   ├── audio
│   │   │   ├── augmented
│   │   │   │   ├── 0_healthy
│   │   │   │   └── 1_alzheimer
│   │   │   └── base
│   │   │       ├── 0_healthy
│   │   │       └── 1_alzheimer
│   │   └── eye_tracking
│   │       ├── augmented
│   │       │   ├── 0_healthy
│   │       │   └── 1_alzheimer
│   │       └── base
│   │           ├── 0_healthy
│   │           └── 1_alzheimer
│   └── text
│       └── base
│           ├── 0_healthy
│           └── 1_alzheimer
└── split
    └── metadata
```

## Configuration of the Experiment

In order to run an experiment, one has to manually edit the configuration JSON files (stored in `params`) related to some core aspects of the analysis. The core aspects of an experiments are the following:

1. **Experimental setting** (`experiment.json`): NN architecture and dataset, parameters related to the the training procedure;
3. **Network** (`networks/`): architecture-specific parameters for the selected NN (or for the submodules in case of multi-modal architectures);
4. **Dataset** (`dataset/`): paths to the data modalities involved in the experiment;
5. **Modality** (`modalities/`):  modality-specific parameters for the modalities handled by the selected NN. 

### Experimental Setting

These parameters define the general setting of the experiments and the configuration for the train procedure.

#### Description of the Parameters

##### General

| Name           | Type  | Values                                                       | Description                                                  |
| -------------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `device`       | `str` | `cpu` or a string matching the regex `\bcuda:\b\d+`          | The device to be used when performing the train procedure. If a selected CUDA device is not available, defaults to CPU. |
| `dataset_name` | `str` | Any dataset name having corresponding parameters in the `params/dataset/` folder | The dataset to be used to train and evaluate the model.      |
| `num_seeds`    | `int` | Any positive integer number                                  | The number of different random seeds for which the CV procedure on each selected split must be run. The random seeds are generated increasing the a base seed (i.e., 0) by one unit at a time (e.g. `num_seed = 3` and implies seeds ranging from 1 to 3 included). |

##### Train

| Name             | Type    | Values                                                       | Description                                                  |
| ---------------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `network_type`   | `str`   | Any network name having corresponding parameters in the `params/networks/` folder | The network to be used to train and evaluate the model.      |
| `epochs`         | `int`   | Any positive integer number                                  | The maximum number of epochs (i.e. iterations through the whole dataset) to be performed. |
| `optimizer`      | `str`   | Any optimiser name implemented in `classes/factories/OptimizerFactory` | The type of optimiser to be used for updating the weights. Currently implemented are `SGD`, `Adam` and `AdamW`. |
| `batch_size`     | `int`   | Any positive integer number                                  | The size of the mini-batches                                 |
| `learning_rate`  | `float` | Any positive float number                                    | The learning rate value the optimiser should initialised to. |
| `evaluate_every` | `int`   | Any positive integer number                                  | The frequency (in terms of epochs) with which the model should be evaluated against the validation set. |
| `log_every`      | `int`   | Any positive integer number                                  | The frequency (in terms of epochs) with which the metrics should be logged on terminal. |

###### Early stopping

| Name            | Type  | Values                                                       | Description                                                  |
| --------------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `patience`      | `int` | Any pos integer number greater than zero                     | The number of iterations for which a non-improvement of the monitored metrics is tolerated. |
| `metrics`       | `str` | Any metrics name from those implemented in `classes/core/Evaluator.py ` | The monitored val metrics (e.g. `auc`, `f1`, `loss`, etc…)   |
| `metrics_trend` | `str` | `increasing` or `decreasing`                                 | Whether the monitored metrics is supposed to increase or decrease (e.g. `auc` should increase while `loss` should decrease). |

###### CV

| Name              | Type   | Values                      | Description                                                  |
| ----------------- | ------ | --------------------------- | ------------------------------------------------------------ |
| `k`               | `int`  | Any positive integer number | The number of folds which the dataset must be split into.    |
| `down_sample`     | `bool` | Any Boolean value           | Whether or not to down_sample the val set to have the same amount of items in the majority class as the train set. |
| `use_cv_metadata` | `bool` | Any Boolean value           | Whether or not to use metadata information for the split instead of generating it anew. |

#### Example

```json
{
  "device": "cuda:3",
  "dataset_name": "confusion",
  "num_seeds": 1,
  "train": {
    "network_type": "rnn",
    "epochs": 100,
    "optimizer": "Adam",
    "criterion": "CrossEntropyLoss",
    "batch_size": 256,
    "learning_rate": 3e-5,
    "log_every": 1,
    "evaluate_every": 1,
    "early_stopping": {
      "patience": 50,
      "metrics": "auc",
      "metrics_trend": "increasing"
    }
  },
  "cv": {
    "k": 10,
    "down_sample_rate": 3,
    "use_cv_metadata": false
  }
}
```

### Dataset

#### Description of the Parameters

##### Classes

| Name       | Type  | Values                          | Description                                                  |
| ---------- | ----- | ------------------------------- | ------------------------------------------------------------ |
| `pos` | `str` | Any string but the empty string | The ID of the pos class. It must be the same as the name of the folder containing the pos data items (without the `1_` prefix). |
| `neg` | `str` | Any string but the empty string | The ID of the neg class. It must be the same as the name of the folder containing the neg data items (without the `0_` prefix). |

##### Paths

| Name               | Type  | Values                          | Description                                                  |
| ------------------ | ----- | ------------------------------- | ------------------------------------------------------------ |
| `dataset_dir`   | `str` | Any string but the empty string | The path to the main datasets folder (containing all the datasets, including the one whose parameters are being set in the present configuration file). |
| `dataset_metadata` | `str` | Any string but the empty string | The path to the folder containing general metadata for the specific dataset. |
| `cv_metadata`      | `str` | Any string but the empty string | The path to the folder containing CV-splits-related metadata for the specific dataset. |

###### Modalities

| Name        | Type  | Values                          | Description                                                  |
| ----------- | ----- | ------------------------------- | ------------------------------------------------------------ |
| `images`    | `str` | Any string but the empty string | The path to the folder containing the data related to the `images` modality. |
| `sequences` | `str` | Any string but the empty string | The path to the folder containing the data related to the `sequences` modality. |
| `text`      | `str` | Any string but the empty string | The path to the folder containing the data related to the `text` modality. |
| `audio`     | `str` | Any string but the empty string | The path to the folder containing the data related to the `audio` modality. |

#### Example

File `alzheimer.json` with respect to the Alzheimer dataset stored at `../dataset/alzheimer`.

```json
{
  "classes": {
    "pos": "alzheimer",
    "neg": "healthy"
  },
  "paths": {
    "dataset_dir": "../dataset/alzheimer",
    "dataset_metadata": "metadata",
    "cv_metadata": "split/metadata/eye_tracking/",
    "modalities": {
      "images": "images",
      "sequences": "sequences",
      "text": "text",
      "audio": "audio"
    }
  }
}
```

### Modalities

#### Sequences

| Name                | Type   | Values                                 | Description                                                  |
| ------------------- | ------ | -------------------------------------- | ------------------------------------------------------------ |
| `path_to_data` | `str`  | Any string but the empty string | The path to the sequences to be loaded from the dataset. This must match with the name of a folder inside `modalities/sequences` in the dataset. |
| `file_format`       | `str`  | Any string but the empty string        | The file format of the data items (e.g. `pkl`, `csv`, etc…). |
| `num_features`      | `int`  | Any positive integer | The number of features to be used. Features are selected sequentially from the features at position 0. |
| `length`        | `int`  | Any positive integer | The target length of the truncated sequences.                |
| `truncate_from`     | `str`  | Either `head` or `tail`                | The starting point for the truncation of the sequence. |

#### Images

| Name           | Type  | Values                          | Description                                                  |
| -------------- | ----- | ------------------------------- | ------------------------------------------------------------ |
| `path_to_data` | `str` | Any string but the empty string | The path to the images to be loaded from the dataset. This must match with the name of a folder inside `modalities/images` in the dataset. |
| `file_format`  | `str` | Any string but the empty string | The file format of the data items (e.g. `png`, `jpg`, etc…). |
| `num_channels` | `int` | Any integer in {1, 2, 3}        | The number of channels to be considered for the input images (e.g. 1 for black-and-white, 3 for RGB, etc…). |

##### Size

| Name     | Type  | Values                   | Description                                           |
| -------- | ----- | ------------------------ | ----------------------------------------------------- |
| `width`  | `int` | Any non-negative integer | The width which the input images will be resized to.  |
| `height` | `int` | Any non-negative integer | The height which the input images will be resized to. |

#### Text

| Name                        | Type  | Values                                                       | Description                                                  |
| --------------------------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `path_to_data`              | `str` | Any string but the empty string                              | The path to the text documents to be loaded from the dataset. This must match with the name of a folder inside `modalities/text` in the dataset. |
| `file_format`               | `str` | Any string but the empty string                              | The file format of the data items (e.g. `txt`).              |
| `max_sentences`             | `int` | The dataset-specific maximum number of sentences in a document | The dataset-specific maximum number of sentences in a document. This parameter is not required for the Transformer models. |
| `max_words`                 | `int` | The dataset-specific maximum number of words in a sentence   | The dataset-specific maximum number of words in a sentence. This parameter is not required for the Transformer models. |
| `vocabulary_size`           | `int` | The dataset-specific size of the vocabulary                  | The dataset-specific size of the vocabulary. This parameter is not required for the Transformer models. |
| `path_to_pretrained_models` | `str` | Any string                                                   | A local path to some pretrained model (e.g. Transformer).    |

## Report of the Experiments

The results of the experiments are saved in a user-named folder, referred to as "report". The information in the report is nested by seed and iteration and include, for each fold:

* Training, validation and test metrics;
* Training, validation and test plots of metrics over epochs;
* Training, validation and test predictions

All the parameters involved in the experiment are dumped into the main folder of the report, namely in three files:

1. `data.json`: data-related parameters such as dataset parameters (including name and main modality used for the experiment);
2. `experiment.json`: general configuration of the experiment included in the `experiment.json` configuration file;
3. `network_params`: network-related parameters and configuration for the corresponding modality.

## Extending the code

### Adding a Dataset

The procedure to add a new dataset is the following:

1. **Creating the dataset ID**. Each new dataset is associated to a unique ID (e.g. Alzheimer >> `alzheimer`) to allow for it to be referenced in the configuration of the experiment. New datasets must be included in the `dataset` folder in a subdirectory named after their IDs (e.g. Alzheimer >> `dataset/alzheimer/...`). Note that case sensitivity matters;
2. **Structuring the dataset**. The dataset must be structured as described in *§ Dataset* in this document;
3. **Defining the grouper**. Each dataset must be coupled with a specific class defining its data grouping policy within the splits for the CV (e.g. group patients by sub-tasks or by cyclic split). This class must be defined at `classes/data/groupers` and must subclass `DataGrouper`. After defining the new grouper class, make sure to update the factory for the groupers at `classes/factories`.

Note that if manually augmented data is used, the corresponding files must be indexed according to and increasing integer value greater than zero.  If for example a sequence stored in a file named L1-5.csv is split in 4 different subsequences, the 4 corresponding files must be named as L1-5-1.csv, L1-5-2.csv, L1-5-3.csv, L1-5-4.csv (the “-” character dividing the item ID and the augmentation index matters).

### Adding a Modality

Each new modality is associated to a unique ID (e.g. sequences >> `sequences`) to allow for it to be referenced in the configuration of the experiment. New modalities must be included in the `dataset/specific_dataset/modalities` folder and may include specific data sources and representations. The procedure to add a new modality is the following:

1. Move the data for the new modality at `dataset/specific_dataset/modalities/data_source[optional]/new_modality/representation[optional]` making sure the folder is structured as described at *§ Dataset* of this document;
2. Write a new data loader at `classes/data/loaders` defining how the data items belonging to the new modality must be loaded (note that the new class must subclass `data.Loader` ;
3. Update the `factories.LoaderFactory` inserting a new key value pair in the corresponding map binding the ID of the modality to its loader;
4. Write a new file at `parameters/modalities` named as the ID of the modality including the modality-specific parameters.

### Adding a New Module

Each module is associated to an ID that allows for it to be referenced in the configuration of the experiment (e.g. Hierarchical Attention Network >> `han`). To add a new network, a new module must be defined. Modules are stored at `classes/modules` and include two classes:

1. The network class defining the NN architecture and its forward phase (i.e. a subclass of `nn.Module`);
2. The model class defining how the network is instantiated an how the inputs are fed to it. This must subclass `classes.core.Model`.

Note that the constructor of the network classes must have the following signature:

```python
def __init__(network_params: dict, activation: bool)
	"""
	@param network_params: dictionary contatining both the parameters listed in the configuration file of the module and the parameters listed in the configuration file of the corresponding modality
    	@param activation: whether or not the architecture will feature classification layers (set to None for implementing some features fusion policies in multimodal networks)
	"""
```

After writing the new module, the following factories must be updated inserting a new key value pair in the corresponding map:

* `factories.ModelFactory`, binding the ID of the new module to the corresponding model;
* `factories.NetworkFactory`, binding the ID of the new module to the corresponding network.

Next, the new module must be bounded to a modality (or to multiple modalities in case of multimodal networks). This can be done updating the corresponding map at `binders.ModalityBinder`.

Finally, the module can be configured either:

* Using a preexisting  JSON file at `params/networks` binding the two together at `binders.ParamsBinder` , or
* Writing a fresh JSON configuration file for the new module

Note that the binding between the configuration file and the module is optional. If no binding exists, the `binders.ParamsBinder` will search for a configuration file named after the ID of the new module. 

### Adding a Criterion

In order to add a new criterion, `factories.CriterionFactory` must be updated inserting a new key value pair in the corresponding map. This binds the ID of the new criterion to its in-place definition.

### Adding an Optimiser

In order to add a new optimiser, `factories.OptimizerFactory` must be updated inserting a new key value pair in the corresponding map. This binds the ID of the new optimiser to its in-place definition.
