import torch
from torch.utils.data import DataLoader

from classifier.classes.core.Evaluator import Evaluator
from classifier.classes.core.Model import Model
from classifier.classes.factories.ModelFactory import ModelFactory
from classifier.classes.utils.Params import Params


class Trainer:

    def __init__(self, network_type: str, training_params: dict, device: torch.device, path_to_best_model: str):
        """
        :param training_params: the training related params in the experiment.json file
        :param device: the device which to run on (gpu or cpu)
        :param path_to_best_model: the path at which the best model is saved during training
        """
        self.__device = device
        self.__network_type = network_type
        self.__path_to_best_model = path_to_best_model

        self.__epochs = training_params["epochs"]
        self.__optimizer_type = training_params["optimizer"]
        self.__criterion_type = training_params["criterion"]
        self.__clip_gradient = training_params["clip_gradient"]
        self.__batch_size = training_params["batch_size"]

        self.__log_every = training_params["early_stopping"]["log_every"]
        self.__evaluate_every = training_params["early_stopping"]["evaluate_every"]
        self.__early_stopping = training_params["early_stopping"]["active"]
        self.__early_stopping_patience = training_params["early_stopping"]["patience"]
        self.__monitored_metric = training_params["early_stopping"]["metrics"]
        self.__monitored_metric_trend = training_params["early_stopping"]["metrics_trend"]
        self.__monitored_metric_best_value = 0.0 if self.__monitored_metric_trend == "increasing" else 1000
        self.__epochs_without_improvement = 0

        self.__learning_rate = training_params["learning_rate"]["initial_value"]
        self.__learning_rate_decay_ratio = training_params["learning_rate"]["decay_ratio"]
        self.__learning_rate_decay_patience = training_params["learning_rate"]["decay_patience"]
        self.__max_learning_rate_decreases = training_params["learning_rate"]["max_decreases"]
        self.__reload_best_on_decay = training_params["learning_rate"]["reload_best_on_decay"]
        self.__num_learning_rate_decreases = 0

        self.__model = self.__create_model()
        self.__evaluator = Evaluator(self.__batch_size, self.__device)

    def get_model(self) -> Model:
        return self.__model

    def __create_model(self) -> Model:
        """
        Instantiates the model for the training
        :return: a specialized model subclassing Model
        """
        network_params = Params.load_network_params(self.__network_type)
        network_params["device"] = self.__device

        model = ModelFactory().get(self.__network_type, network_params)
        model.print_model_overview()
        model.set_optimizer(self.__optimizer_type, self.__learning_rate, self.__clip_gradient)
        model.set_criterion(self.__criterion_type)

        return model

    def train(self, data: dict) -> dict:
        """
        Trains the model according to the established parameters and the given data
        :param data: a dictionary of data loaders containing training, validation and test data
        :return: the evaluation metrics of the training
        """
        print("\n Training the model... \n")

        metrics = {"training": [], "validation": [], "test": []}

        for epoch in range(self.__epochs):
            print("\n *** Epoch {}/{} *** \n".format(epoch + 1, self.__epochs))

            self.__model.train_mode()
            self.__train_epoch(data["training"], epoch)

            if not (epoch + 1) % self.__evaluate_every:
                evaluation = self.__evaluator.evaluate_model(data, self.__model)
                metrics["training"] += [evaluation["training"]["metrics"]]
                metrics["validation"] += [evaluation["validation"]["metrics"]]
                metrics["test"] += [evaluation["test"]["metrics"]]

                if self.__early_stopping:
                    self.__update_early_stopping_metrics(metrics["validation"][-1][self.__monitored_metric])
                    if self.__epochs_without_improvement == self.__early_stopping_patience:
                        print("\n ** No decrease in validation {} for {} evaluations "
                              "(evaluation frequency is every {} epochs). Stopping training early ** \n"
                              .format(self.__monitored_metric, self.__early_stopping_patience, self.__evaluate_every))
                        break

                if ((self.__max_learning_rate_decreases > 0 and self.__epochs_without_improvement > 0) and
                        (self.__epochs_without_improvement % self.__learning_rate_decay_patience == 0) and
                        (self.__num_learning_rate_decreases < self.__max_learning_rate_decreases)):
                    self.__decay_learning_rate()

        print("\n Finished training! \n")

        return metrics

    def __train_epoch(self, dataloader: DataLoader, epoch: int):
        """
        Trains the model batch-wise for the current epoch
        :param dataloader: the data which to train the model on
        :param epoch: the number identifying the current epoch
        """
        running_loss, running_accuracy = 0.0, 0.0
        num_batches = len(dataloader)

        for i, (x, y) in enumerate(dataloader):

            self.__model.reset_gradient()

            y = y.long().to(self.__device)
            o = self.__model.predict(x).to(self.__device)

            running_loss += self.__model.update_weights(o, y)
            running_accuracy += Evaluator.batch_accuracy(o, y)

            if not (i + 1) % self.__log_every:
                avg_loss, avg_accuracy = running_loss / self.__log_every, running_accuracy / self.__log_every
                print("[ Epoch: {}/{}, batch: {}/{} ] [ Loss: {:.4f} | Accuracy: {:.4f} ]"
                      .format(epoch + 1, self.__epochs, i + 1, num_batches, avg_loss, avg_accuracy))
                running_loss, running_accuracy = 0.0, 0.0

        print("\n ........................................................... \n")

    def __update_early_stopping_metrics(self, monitored_metrics: float):
        """
        Decides whether or not to early stop the training based on the early stopping conditions
        :param monitored_metrics: the monitored validation metrics (e.g. AUC, loss)
        """
        if self.__monitored_metric_trend == "increasing":
            metrics_check = monitored_metrics > self.__monitored_metric_best_value
        else:
            metrics_check = monitored_metrics < self.__monitored_metric_best_value

        if metrics_check:
            print("\n\t Old best validation {m}: {o} | New best {m}: {n} \n"
                  .format(m=self.__monitored_metric, o=self.__monitored_metric_best_value, n=monitored_metrics))
            print("\t Saving new best model...")
            self.__model.save(self.__path_to_best_model)
            self.__monitored_metric_best_value = monitored_metrics
            self.__epochs_without_improvement = 0
        else:
            self.__epochs_without_improvement += 1

        print("\n Epochs without improvement: ", self.__epochs_without_improvement)
        print("\n ........................................................... \n")

    def __decay_learning_rate(self):
        """
        Decays the learning rate according to the given decay ratio and reloads the latest best model
        """
        print("\n ** No improvement in validation {} in {} epochs. Reducing learning rate. ** \n"
              .format(self.__monitored_metric, self.__epochs_without_improvement))

        print("\n\t - Old learning rate: ", self.__learning_rate)
        self.__learning_rate *= self.__learning_rate_decay_ratio
        print("\n\t - New learning rate: ", self.__learning_rate)

        self.__num_learning_rate_decreases += 1

        if self.__reload_best_on_decay:
            self.__model.load(self.__path_to_best_model)

        self.__model.set_optimizer(self.__optimizer_type, self.__learning_rate, self.__clip_gradient)
