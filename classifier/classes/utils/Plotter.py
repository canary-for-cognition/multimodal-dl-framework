import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


class Plotter:

    def __init__(self, path_to_plots: str):
        self.plots_folder = path_to_plots
        self.fold = 0
        self.range = []

        self.color_1, self.color_1_accent = "#4285F4", "#C9F9FF"
        self.color_2, self.color_2_accent = "#DB4437", "#FDDBC3"
        self.color_3, self.color_3_accent = "#0F9D58", "#92F5BF"

    def __set_range(self, num_epochs: int):
        self.range = range(0, num_epochs)

    def __set_fold(self, fold: int):
        self.fold = fold

    @staticmethod
    def __fetch_metrics(metrics: dict, metric_type: str, data_type: str):
        return [epoch_metrics[metric_type] for epoch_metrics in metrics[data_type]]

    def plot_metrics(self, metrics: dict, fold: int):
        self.__set_fold(fold)
        self.__set_range(len(metrics["training"]))

        for metric in ["loss", "accuracy", "f1", "auc"]:
            self.plot_metric(self.__fetch_metrics(metrics, metric_type=metric, data_type="training"),
                             self.__fetch_metrics(metrics, metric_type=metric, data_type="validation"),
                             self.__fetch_metrics(metrics, metric_type=metric, data_type="test"),
                             metric=metric)

        sensitivity = {
            "training": self.__fetch_metrics(metrics, metric_type="sensitivity", data_type="training"),
            "validation": self.__fetch_metrics(metrics, metric_type="sensitivity", data_type="validation"),
            "test": self.__fetch_metrics(metrics, metric_type="sensitivity", data_type="test")
        }

        specificity = {
            "training": self.__fetch_metrics(metrics, metric_type="specificity", data_type="training"),
            "validation": self.__fetch_metrics(metrics, metric_type="specificity", data_type="validation"),
            "test": self.__fetch_metrics(metrics, metric_type="specificity", data_type="test")
        }

        self.plot_sensitivity_vs_specificity(sensitivity, specificity)

    def plot_sensitivity_vs_specificity(self, sensitivity: dict, specificity: dict):
        fig, ax = plt.subplots()

        ax.plot(self.range, sensitivity["training"], self.color_1, label="training sensitivity")
        ax.plot(self.range, specificity["training"], self.color_1_accent, label="training specificity")

        ax.plot(self.range, sensitivity["validation"], self.color_2, label="validation sensitivity")
        ax.plot(self.range, specificity["validation"], self.color_2_accent, label="validation specificity")

        ax.plot(self.range, sensitivity["test"], self.color_3, label="test sensitivity")
        ax.plot(self.range, specificity["test"], self.color_3_accent, label="test specificity")

        ax.set(xlabel="Epochs", ylabel="Sensibility / Specificity", title="Sensibility and Specificity")
        ax.legend()

        fig.savefig(os.path.join(self.plots_folder, "fold_" + str(self.fold) + "_sensibility_specificity.png"))
        plt.close(fig)

    def plot_metric(self, training: list, validation: list, test: list, metric: str):
        fig, ax = plt.subplots()

        ax.plot(self.range, training, self.color_1, label="training")
        ax.plot(self.range, validation, self.color_2, label="validation")
        ax.plot(self.range, test, self.color_3, label="test")

        ax.set(xlabel="Epochs", ylabel=metric.capitalize(), title=metric.capitalize())
        ax.legend()

        fig.savefig(os.path.join(self.plots_folder, "fold_" + str(self.fold) + "_" + metric + ".png"))
        plt.close(fig)
