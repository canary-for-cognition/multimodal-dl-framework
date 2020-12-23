import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import recall_score, f1_score, roc_auc_score, roc_curve
from tqdm import tqdm

from classifier.classes.core.Model import Model


class Evaluator:

    def __init__(self, device: torch.device):
        """
        :param device: the device which to run on (gpu or cpu)
        """
        self.__device = device

    def evaluate_model(self, data: dict, model: Model, path_to_model: str = "") -> dict:
        """
        Evaluates the saved best model against train, val and test data
        :param data: a dictionary tuple containing the data loaders for train, val and test
        :param model: the model to be evaluated
        :param path_to_model: the path to the saved serialization of the best model
        :return: the evaluation of the model on train, val and test data, including metrics, gt and preds
        """
        model.evaluation_mode()

        if path_to_model != "":
            model.load(path_to_model)

        evaluation = {}

        for set_type, dataloader in data.items():

            loss, accuracy, y_scores, y_true = [], [], [], []

            with torch.no_grad():

                for i, (x, y) in tqdm(enumerate(dataloader), desc="Evaluating {}".format(set_type)):
                    x = x.float().to(self.__device)
                    y = y.long().to(self.__device)
                    o = model.predict(x).to(self.__device)

                    loss += [model.get_loss(o, y)]
                    accuracy += [self.batch_accuracy(o, y)]

                    y_scores += torch.exp(o).cpu().numpy().tolist()
                    y_true += y.cpu().numpy().tolist()

            y_scores, y_true = np.array(y_scores).reshape((len(y_scores), 2)), np.array(y_true)
            y_pred = np.array((y_scores[:, 1] >= self.__optimal_roc_threshold(y_true, y_scores[:, 1])), dtype=np.int)
            metrics = self.__compute_metrics(y_true, y_pred, y_1_scores=y_scores[:, 1])
            metrics["accuracy"], metrics["loss"] = np.mean(accuracy), np.mean(loss)

            print("\n {} metrics: \n".format(set_type.upper()))
            for metric, value in metrics.items():
                print(("\t - {} " + "".join(["."] * (15 - len(metric))) + " : {:.4f}").format(metric, value))

            evaluation[set_type] = {"metrics": metrics, "gt": y_true, "preds": y_pred}

        return evaluation

    @staticmethod
    def __compute_metrics(y_true: np.array, y_pred: np.array, y_1_scores: np.array) -> dict:
        """
        Computes the metrics for the given preds and labels
        :param y_true: the ground-truth labels
        :param y_pred: the preds of the model
        :param y_1_scores: the probabilities for the pos class
        :return: the following metrics in a dict: Sensitivity (TP rate) / Specificity (FP rate) / Combined  F1 / AUC
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
    def batch_accuracy(o: torch.Tensor, y: torch.Tensor) -> float:
        """
        Computes the accuracy of the preds over the items in a single batch
        :param o: the logit output of datum in the batch
        :param y: the correct class index of each datum
        :return the percentage of correct preds as a value in [0,1]
        """
        corrects = torch.sum((torch.max(o, 1)[1].view(y.size()).data == y.data).long())
        accuracy = corrects / y.size()[0]
        return accuracy.item()

    @staticmethod
    def __optimal_roc_threshold(y_true: np.array, y_1_scores: np.array) -> float:
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
