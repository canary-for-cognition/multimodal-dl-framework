# Pretrained CNN

> Reference: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

Pretrained CNNs feature a model which has previously been trained on a very large dataset to either extract useful features from the input or perform standard classification. In the former case, the parameters of the pretrained model will not be updated during the training procedure.

## Supported pretrained models

> Reference: https://pytorch.org/docs/stable/torchvision/models.html

The following models imported from `torchvision.models` are supported. 

| Model name | Model ID     | Required input size (W x H) |
| ---------- | ------------ | --------------------------- |
| ResNet     | `resnet`     | 224 x 224                   |
| AlexNet    | `alexnet`    | 224 x 224                   |
| VGG        | `vgg`        | 224 x 224                   |
| SqueezeNet | `squeezenet` | 224 x 224                   |
| DenseNet   | `densenet`   | 224 x 224                   |
| Inception  | `inception`  | 299 x 299                   |

### Adding support for a new model

In order to add support for a new model, it is sufficient to update the `pre_trained_models_map` in the method `PretrainedCNN.__select_pre_trained_model` at `PretrainedCNN.py`.  This map binds a model ID that can be specified in the experiment configuration to the respective class in `torchvision.models`. 

Subsequently, the `initializations_map` must be updated with a function specifying how the classification layers are added on top of the pretrained architecture in the `PretrainedCNN.__add_classifier` method at `PretrainedCNN.py`.

## Configuration

| Name                  | Type   | Values                                                       | Description                                                  |
| --------------------- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `output_size`         | `int`  | Any positive integer greater than zero                       | The number of labels for the classification problem.         |
| `features_extraction` | `bool` | Any Boolean value                                            | Whether or not to use the pretrained model for features extraction. |
| `pre_trained_mode`    | `str`  | Any string included in the table of supported pretrained models | The pretrained model to be used.                             |