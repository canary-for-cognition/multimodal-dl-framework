from typing import Dict, Tuple, Callable

import numpy as np
import torch
from torch import nn


class CNN(nn.Module):

    def __init__(self, network_params: Dict, activation: bool = True):
        super().__init__()

        input_size = network_params["input_size"]
        output_size = network_params["output_size"]
        self.__num_conv_blocks = network_params["num_conv_blocks"]
        self.__num_conv = network_params["num_conv"]
        self.__conv_block_params = network_params["layers"]["conv_block"]
        classifier_params = network_params["layers"]["classifier"]
        bottleneck_output_size = classifier_params["linear_1"]["out_features"]
        self.__activation = activation

        conv_blocks, last_conv_out_channels = self._generate_conv_blocks()
        self.conv_and_pool_layers = nn.Sequential(*conv_blocks)

        self.__linear_size = self._compute_linear_size(input_size, output_size=last_conv_out_channels)

        self.fc = nn.Sequential(
            nn.Linear(self.__linear_size, bottleneck_output_size),
            nn.ReLU()
        )

        if self.__activation:
            self.classifier = self.__generate_classifier(classifier_params, output_size)

    @staticmethod
    def __generate_classifier(params: Dict, output_size: int) -> nn.Sequential:
        """
        Generates the layers for the classifier as specified in the configuration of the network
        :return: the generated layers as a sequential model
        """
        layers_names = params.keys()
        out_features = [params[layer_name]["out_features"] for layer_name in layers_names]

        linear_layers = []
        for i in range(len(out_features)):
            if i + 1 < len(out_features):
                linear_layers.append(nn.Linear(out_features[i], out_features[i + 1]))
                linear_layers.append(nn.ReLU())
            else:
                linear_layers.append(nn.Linear(out_features[i], output_size))

        return nn.Sequential(*linear_layers)

    @staticmethod
    def __compute_layer_out_dim(input_dim: int, layer_params: Dict) -> int:
        """
        Computes the output size of a layer of the CNN (convolutional or pooling) as: [(W − K + 2P) / S] + 1
        Where:
            * W is the input volume
            * K is the Kernel size
            * P is the padding
            * S is the stride
        References:
            - https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
            - https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html
        :param input_dim: the size of the input of the layer
        :param layer_params: kernel size, stride and padding for the layer
        :return the output size of a layer of the CNN
        """
        kernel_size = layer_params["kernel_size"]
        stride = layer_params["stride"]
        padding = layer_params["padding"]
        return np.floor(((input_dim + 2 * padding - kernel_size) / stride) + 1)

    def __compute_conv_blocks_out_dim(self, input_dim: int) -> int:
        """
        Computes the output size of the convolutional blocks with respect to the input dimension
        :param input_dim: one of the spacial dimensions of the input
        :return: the output size of the convolutional blocks with respect to the input dimension
        """
        output_dim = input_dim
        for i in range(self.__num_conv_blocks):
            for j in range(self.__num_conv):
                output_dim = self.__compute_layer_out_dim(output_dim, self.__conv_block_params["conv_" + str(j + 1)])
            output_dim = self.__compute_layer_out_dim(output_dim, self.__conv_block_params["pool"])
        return int(output_dim)

    def _compute_linear_size(self, input_size: Tuple, output_size: int) -> int:
        """
        Computes the input size for the fully connected layer
        :param input_size: the size of the input (W x H in case of images). If an element of the tuple is equal to -1,
        then it will be ignored in the computation
        :param output_size: the number of output channels of the last convolutional layer
        :return: the input size for the fully connected layer
        """
        output_dim = np.prod([self.__compute_conv_blocks_out_dim(i) if i != -1 else 1 for i in input_size])
        return output_dim * output_size

    @staticmethod
    def __select_max_pool_type(max_pool_type: str) -> Callable:
        """
        Fetches the callable max pooling layer from a map of supported layers type (e.g. "MaxPool1d", etc...)
        :param max_pool_type: the desired type of max pooling layer (e.g. max_pool_type = 1d -> nn.MaxPool1d)
        :return: the callable max pooling layer fetched from a map of supported types of layers (if present)
        """
        max_pool_types_map = {
            "1d": nn.MaxPool1d,
            "2d": nn.MaxPool2d,
            "3d": nn.MaxPool3d
        }

        if max_pool_type not in max_pool_types_map.keys():
            raise ValueError("Max pooling type {} not supported!".format(max_pool_type))

        return max_pool_types_map[max_pool_type]

    @staticmethod
    def __select_conv_type(conv_type: str) -> Callable:
        """
        Fetches the callable convolutional layer from a map of supported layers type (e.g. "Conv1d", "Conv2d", etc...)
        :param conv_type: the desired type of convolutional layer (e.g. conv_type = 1d -> nn.Conv1d)
        :return: the callable convolutional layer fetched from a map of supported types of layer (if present)
        """
        conv_types_map = {
            "1d": nn.Conv1d,
            "2d": nn.Conv2d,
            "3d": nn.Conv3d
        }

        if conv_type not in conv_types_map.keys():
            raise ValueError("Convolution type {} not supported!".format(conv_type))

        return conv_types_map[conv_type]

    def _generate_conv_blocks(self) -> Tuple:
        """
        Generates num_conv_block convolutional layers with num_conv convolutional layers each
        followed by a max pooling layer
        :return: a list of layers and the output channels of the last layer
        """
        max_pool = self.__select_max_pool_type(self.__conv_block_params["pool"]["type"])
        pooling_kernel_size = self.__conv_block_params["pool"]["kernel_size"]
        pooling_stride = self.__conv_block_params["pool"]["stride"]

        conv_blocks = []
        for i in range(self.__num_conv):
            layer_params = self.__conv_block_params["conv_" + str(i + 1)]
            conv_layer = self.__select_conv_type(layer_params["type"])
            in_channels, out_channels = layer_params["in_channels"], layer_params["out_channels"]
            kernel_size, stride = layer_params["kernel_size"], layer_params["stride"]
            conv_blocks.append(conv_layer(in_channels, out_channels, kernel_size, stride))
            conv_blocks.append(nn.ReLU())
        conv_blocks.append(max_pool(pooling_kernel_size, pooling_stride))

        last_conv_params = self.__conv_block_params["conv_" + str(self.__num_conv)]
        last_conv_out_channels = last_conv_params["out_channels"]
        for _ in range(self.__num_conv_blocks - 1):
            for i in range(self.__num_conv):
                layer_params = self.__conv_block_params["conv_" + str(i + 1)]
                conv_layer = self.__select_conv_type(layer_params["type"])
                kernel_size, stride = layer_params["kernel_size"], layer_params["stride"]
                conv_blocks.append(conv_layer(last_conv_out_channels, last_conv_out_channels * 2, kernel_size, stride))
                conv_blocks.append(nn.ReLU())
                last_conv_out_channels *= 2
            conv_blocks.append(max_pool(pooling_kernel_size, pooling_stride))

        return conv_blocks, last_conv_out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_and_pool_layers(x)
        x = self.fc(x.view(-1, self.__linear_size))
        return self.classifier(x) if self.__activation else x
