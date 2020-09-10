# Transformers

> Reference: https://huggingface.co/transformers/

## Downloading a pretrained model

In source code of the model configuration classes (e.g. [BertConfig](https://huggingface.co/transformers/_modules/transformers/configuration_bert.html#BertConfig), [RobertaConfig](https://huggingface.co/transformers/_modules/transformers/configuration_roberta.html#RobertaConfig), etc…) from the GitHub repository [huggingface/transformers](https://github.com/huggingface/transformers), you can find these URLs such as the following:

```python
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
```

Now follow these steps:

1. Download the model you need by the URL and unzip it, then you will get `[base_model_name]_config.json` (e.g. `bert_config.json`) and `pytorch_model.bin`
2. Rename `[base_model_name]_config.json` to `config.json`
3. Put `config.json` and `pytorch_model.bin` in a folder named `[model_name]` (e.g. `bert-base-uncased`)

Now the local model will be loaded as:

```python
model = BertModel.from_pretrained('path_to_bert_base_uncased')
```

## Supported pretrained architectures

| Architecture name | Architecture ID | Reference                                                  | Pretrained models                                            |
| ----------------- | --------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| BERT              | `bert`          | https://huggingface.co/transformers/model_doc/bert.html    | https://huggingface.co/transformers/_modules/transformers/configuration_bert.html#BertConfig |
| RoBERTa           | `roberta`       | https://huggingface.co/transformers/model_doc/roberta.html | https://huggingface.co/transformers/_modules/transformers/configuration_roberta.html#RobertaConfig |

### Adding support for a new model

In order to add support for a new model, it is sufficient to update the `transformers_map` in the function`Tranformer.__select_pre_trained_architecture` at `Transformer.py`.  This map binds a model ID that can be specified in the experiment configuration to the respective class in the Hugging Face framework. 

## Configuration

| Name                           | Type    | Values                                                      | Description                                                  |
| ------------------------------ | ------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| `pre_trained_architecture`     | `str`   | The ID for any supported architecture                       | The transformer architecture to be used.                     |
| `pre_trained_model`            | `str`   | The ID of any supported model for the selected architecture | The pretrained model to be used along with the selected architecture. |
| `load_local_pre_trained_model` | `bool`  | Any Boolean value                                           | Whether or not to load a locally downloaded model. If set to `false`, the pretrained model will be downloaded (this make take much time). |
| `dropout`                      | `float` | Any non-negative float number                               | The probability of dropping out units during the training process. |
| `output_size`                  | `int`   | Any positive integer greater than zero                      | The number of labels for the classification problem.         |