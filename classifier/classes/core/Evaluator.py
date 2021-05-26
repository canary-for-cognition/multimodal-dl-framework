from typing import Dict

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import recall_score, f1_score, roc_auc_score, roc_curve

from classifier.classes.core.Model import Model


class Evaluator:

    def __init__(self, device: torch.device):
        """
        :param device: the device which to run on (gpu or cpu)
        """
        self.__device = device

    def evaluate(self, data: Dict, model: Model, path_to_model: str = "") -> Dict:
        """
        Evaluates the saved best model against train, val and test data
        :param data: a dictionary tuple containing the data loaders for train, val and test
        :param model: the model to be evaluated
        :param path_to_model: the path to the saved serialization of the best model
        :return: the eval of the model on train, val and test data, including metrics, gt and preds
        """
        model.evaluation_mode()

        if path_to_model != "":
            model.load(path_to_model)

        metrics, gt, preds = {}, {}, {}

        for set_type, dataloader in data.items():

            loss, accuracy, y_scores, y_true = [], [], [], []

            with torch.no_grad():

                for i, (x, y) in enumerate(dataloader):
                    y = y.long().to(self.__device)
                    o = model.predict(x).to(self.__device)

                    loss += [model.get_loss(o, y)]
                    accuracy += [self.batch_accuracy(o, y)]

                    y_scores += torch.exp(o).cpu().numpy().tolist()
                    y_true += y.cpu().numpy().tolist()

            y_scores, y_true = np.array(y_scores).reshape((len(y_scores), 2)), np.array(y_true)
            y_pred = np.array((y_scores[:, 1] >= self.__optimal_roc_threshold(y_true, y_scores[:, 1])), dtype=np.int)
            set_metrics = self.__compute_metrics(y_true, y_pred, y_1_scores=y_scores[:, 1])
            set_metrics["accuracy"], set_metrics["loss"] = np.mean(accuracy), np.mean(loss)

            print("\n {} metrics: \n".format(set_type.upper()))
            for metric, value in set_metrics.items():
                print(("\t - {} " + "".join(["."] * (15 - len(metric))) + " : {:.4f}").format(metric, value))

            metrics[set_type], gt[set_type], preds[set_type] = set_metrics, y_true, y_pred

        return {"metrics": metrics, "gt": gt, "preds": preds}

    @staticmethod
    def batch_accuracy(o: torch.Tensor, y: torch.Tensor) -> float:
        """
        Computes the accuracy of the preds over the items in a single batch
        :param o: the logit output of datum in the batch
        :param y: the correct class index of each datum
        :return the percentage of correct preds as a value in [0,1]
        """
        corrects = torch.sum(torch.max(o, 1)[1].view(y.size()).data == y.data)
        accuracy = corrects / y.size()[0]
        return accuracy.item()

    @staticmethod
    def __compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_1_scores: np.ndarray) -> Dict:
        """
        Computes the metrics for the given preds and labels
        :param y_true: the ground-truth labels
        :param y_pred: the preds of the model
        :param y_1_scores: the probabilities for the pos class
        :return: the following metrics in a Dict: Sensitivity (TP rate) / Specificity (FP rate) / Combined  F1 / AUC
        """
        sensitivity = recall_score(y_true, y_pred, pos_label=1)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        return {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "combined": (sensitivity + specificity) / 2,
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_1_scores)
        }

    @staticmethod
    def __optimal_roc_threshold(y_true: np.ndarray, y_1_scores: np.ndarray) -> float:
        """
        Computes the optimal ROC threshold (defined for more than one sample)
        :param y_true: the ground truth
        :param y_1_scores: the scores for the pos class
        :return: the optimal ROC threshold
        """
        fp_rates, tp_rates, thresholds = roc_curve(y_true, y_1_scores)
        best_threshold, dist = 0.5, 100

        for i, threshold in enumerate(thresholds):
            current_dist = np.sqrt((np.power(1 - tp_rates[i], 2)) + (np.power(fp_rates[i], 2)))
            if current_dist <= dist:
                best_threshold, dist = threshold, current_dist

        return best_threshold
