import os

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import recall_score, f1_score, roc_auc_score, roc_curve
from tqdm import tqdm

from classifier.classes.core.Model import Model


class Evaluator:

    def __init__(self, batch_size: int, device: torch.device):
        """
        :param batch_size: the size of the batch of data fed to the model
        :param device: the device which to run on (gpu or cpu)
        """
        self.__model = None
        self.__device = device
        self.__batch_size = batch_size

    def evaluate_model(self, data: dict, model: Model):
        """
        Evaluates the given model against training, validation and test data
        :param data: a dictionary containing the data loaders for training, validation and test
        :param model: the model to be evaluated
        :return: the evaluation of the model on training, validation and test data
        """
        print(" Evaluating model...")
        self.__model = model
        self.__model.evaluation_mode()
        return self.__evaluate_all_data(data)

    def evaluate_saved_model(self, data: dict, model: Model, path_to_best_model: str) -> dict:
        """
        Evaluates the saved best model against training, validation and test data
        :param data: a dictionary tuple containing the data loaders for training, validation and test
        :param model: the model to be evaluated
        :param path_to_best_model: the path to the saved serialization of the best model
        :return: the evaluation of the model on training, validation and test data
        """
        print(" Evaluating saved model...")
        self.__model = model
        self.__model.load(path_to_best_model)
        self.__model.evaluation_mode()
        return self.__evaluate_all_data(data)

    def __evaluate_all_data(self, data: dict):
        """
        Evaluates the model against training, validation and test data
        :param data: a dictionary containing the data loaders for training, validation and test
        :return: the evaluation of the model on training, validation and test data
        """
        return {
            "training": self.__evaluate_data(data["training"], mode="training"),
            "validation": self.__evaluate_data(data["validation"], mode="validation"),
            "test": self.__evaluate_data(data["test"], mode="test")
        }

    def evaluate_predictions(self, predictions: list) -> dict:
        """
        Computes the metrics for the given predictions
        :param predictions: a list of predictions including ground-truth and probability scores
        :return: the following metrics in a dict:
            * Sensitivity (TP rate) / Specificity (FP rate) / Combined
            * Accuracy / F1 / AUC
        """
        print("\n Evaluating test predictions... \n")
        y_true = np.array([labels for prediction in predictions for labels in prediction["y_true"]])
        y_1_scores = np.array([scores for prediction in predictions for scores in prediction["y_scores"]])[:, 1]
        threshold = self.__compute_optimal_roc_threshold(y_true, y_1_scores)
        y_pred = np.array((y_1_scores >= threshold), dtype=np.int)
        return self.__compute_metrics(y_true, y_pred, y_1_scores)

    def __evaluate_batches(self, data_loader: torch.utils.data.DataLoader, mode: str) -> tuple:
        """
        Evaluates the input data batch by batch
        :param data_loader: the data which to run the evaluation on
        :param mode: the type of the incoming data (in {training, validation, test})
        :return the predictions of the model and the evaluation loss
        """
        items_ids, y_scores, y_true = [], [], []
        running_accuracy, running_loss = 0.0, 0.0

        with torch.no_grad():
            for i, (inputs, labels) in tqdm(enumerate(data_loader), desc="Evaluating {}".format(mode)):
                labels = labels.long().to(self.__device)
                outputs = self.__model.predict(inputs, ).to(self.__device)

                running_loss += self.__model.compute_loss(outputs, labels)
                running_accuracy += self.compute_batch_accuracy(outputs, labels)

                items_ids += self.__get_items_ids(data_loader, mode, batch_index=i)
                y_scores += torch.exp(outputs).cpu().numpy().tolist()
                y_true += labels.cpu().numpy().tolist()

        loss, accuracy = running_loss / len(data_loader), running_accuracy / len(data_loader)
        print("\n [ Average {} scores ] loss: {:.5f} | accuracy: {:.5f}".format(mode, loss, accuracy))

        predictions = {
            "items_ids": items_ids,
            "y_scores": np.array(y_scores).reshape((len(y_scores), 2)),
            "y_true": np.array(y_true)
        }

        return predictions, loss

    @staticmethod
    def __compute_metrics(y_true: np.array, y_pred: np.array, y_1_scores: np.array) -> dict:
        """
        Computes the metrics for the given predictions and labels
        :param y_true: the ground-truth labels
        :param y_pred: the predictions of the model
        :param y_1_scores: the probabilities for the positive class
        :return: the following metrics in a dict:
            * Sensitivity (TP rate) / Specificity (FP rate) / Combined
            * Accuracy / F1 / AUC
        """
        metrics = {}

        if len(np.unique(y_true)) > 1:
            sensitivity = recall_score(y_true, y_pred, pos_label=1)
            specificity = recall_score(y_true, y_pred, pos_label=0)

            metrics = {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "combined": (sensitivity + specificity) / 2,
                "f1": f1_score(y_true, y_pred),
                "auc": roc_auc_score(y_true, y_1_scores)
            }

        correct = np.array(y_true == y_pred).sum()
        metrics["accuracy"] = correct / len(y_true)

        return metrics

    def __evaluate_data(self, data_loader: torch.utils.data.DataLoader, mode: str) -> dict:
        """
        Evaluate a given data and return predictions and metrics in a dict
        :param data_loader: the data over which metric are calculated
        :param mode: the type of the incoming data (training, validation or test)
        :return the predictions and the following metrics in a dict:
            * Sensitivity (TP rate) / Specificity (FP rate) / Combined
            * Accuracy / F1 / AUC
            * Loss
        """
        predictions, loss = self.__evaluate_batches(data_loader, mode)

        threshold = self.__compute_optimal_roc_threshold(predictions["y_true"], predictions["y_scores"][:, 1])
        predictions["y_pred"] = np.array((predictions["y_scores"][:, 1] >= threshold), dtype=np.int)

        metrics = self.__compute_metrics(predictions["y_true"], predictions["y_pred"], predictions["y_scores"][:, 1])
        metrics["loss"] = loss

        print("\n {} metrics: \n".format(mode.capitalize()))
        for metric, value in metrics.items():
            print(("\t - {} " + "".join(["."] * (15 - len(metric))) + " : {}").format(metric, value))

        return {"metrics": metrics, "predictions": predictions}

    def __get_items_ids(self, data_loader: torch.utils.data.DataLoader, mode: str, batch_index: int) -> list:
        """
        Fetches the ids of the items of a given batch. If mode is training, the ids are not available since the
        data is shuffled and thus the index of the item in the batch does not match the one in the data loader
        :param data_loader: a data
        :param mode: the type of the data (in {training, validation, test})
        :param batch_index: the index of the batch (in [0, dataset_size / batch_size))
        :return: the ids of the items in the batch (if mode is not training, else placeholder ids)
        """
        num_samples = len(data_loader.dataset.samples)
        items_indices = range(batch_index * self.__batch_size, min((batch_index + 1) * self.__batch_size, num_samples))
        if mode == "training":
            return [j for j in items_indices]
        else:
            return [data_loader.dataset.samples[j][0].split(os.sep)[-1].split(".")[0] for j in items_indices]

    @staticmethod
    def compute_batch_accuracy(logit: torch.Tensor, target: torch.Tensor) -> float:
        """
        Computes the accuracy of the predictions over the items in a single batch
        :param logit: the logit output of datum in the batch
        :param target: the correct class index of each datum
        :return the percentage of correct predictions as a value in [0,1]
        """
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / target.size()[0]
        return accuracy.item()

    @staticmethod
    def __compute_optimal_roc_threshold(y_true: np.array, y_1_scores: np.array) -> float:
        """
        Computes the optimal ROC threshold
        :param y_true: the ground truth
        :param y_1_scores: the scores for the positive class
        :return: the optimal ROC threshold (defined for more than one sample, else 0.5)
        """
        if len(np.unique(y_true)) < 2:
            return 0.5

        fp_rates, tp_rates, thresholds = roc_curve(y_true, y_1_scores)
        best_threshold, dist = 0.5, 100

        for i, threshold in enumerate(thresholds):
            current_dist = np.sqrt((np.power(1 - tp_rates[i], 2)) + (np.power(fp_rates[i], 2)))
            if current_dist <= dist:
                best_threshold, dist = threshold, current_dist

        return best_threshold
