from classifier.classes.core.Evaluator import Evaluator
from classifier.classes.core.Model import Model
from classifier.classes.factories.ModelFactory import ModelFactory
from classifier.classes.utils.Params import Params


class Trainer:

    def __init__(self, train_params: dict, path_to_best_model: str):
        """
        :param train_params: the train related params in the experiment.json file
        :param path_to_best_model: the path at which the best model is saved during train
        """
        self.__device = train_params["device"]
        self.__path_to_best_model = path_to_best_model
        self.__epochs = train_params["epochs"]
        self.__optimizer_type = train_params["optimizer"]

        self.__log_every = train_params["early_stopping"]["log_every"]
        self.__evaluate_every = train_params["early_stopping"]["evaluate_every"]
        self.__patience = train_params["early_stopping"]["patience"]
        self.__es_metric = train_params["early_stopping"]["metrics"]
        self.__es_metric_trend = train_params["early_stopping"]["metrics_trend"]
        self.__es_metric_best_value = 0.0 if self.__es_metric_trend == "increasing" else 1000
        self.__epochs_no_improvement = 0

        self.__lr = train_params["learning_rate"]["initial_value"]
        self.__lr_decay_ratio = train_params["learning_rate"]["decay_ratio"]
        self.__lr_decay_patience = train_params["learning_rate"]["decay_patience"]
        self.__max_lr_decreases = train_params["learning_rate"]["max_decreases"]
        self.__num_lr_decreases = 0

        self.__model = self.__create_model(train_params["network_type"], train_params["criterion"])
        self.__evaluator = Evaluator(self.__device)

    def __create_model(self, network_type: str, criterion_type: str) -> Model:
        """
        Instantiates the model for the train
        @param network_type: the type of network to be created as specified in the params of the experiment
        @param criterion_type: the type of criterion to be used to evaluate the error (i.e. compute the loss)
        :return: a specialized model subclassing Model
        """
        network_params = Params.load_network_params(network_type)
        network_params["device"] = self.__device

        model = ModelFactory().get(network_type, network_params)
        model.print_model_overview()
        model.set_optimizer(self.__optimizer_type, self.__lr)
        model.set_criterion(criterion_type)

        return model

    def train(self, data: dict) -> tuple:
        """
        Trains the model according to the established parameters and the given data
        :param data: a dictionary of data loaders containing train, val and test data
        :return: the evaluation metrics of the training and the trained model
        """
        print("\n Training the model... \n")

        metrics = {"train": [], "val": [], "test": []}
        training_loader = data["train"]

        for epoch in range(self.__epochs):
            print("\n *** Epoch {}/{} *** \n".format(epoch + 1, self.__epochs))

            self.__model.train_mode()

            running_loss, running_accuracy = 0.0, 0.0
            num_batches = len(training_loader)

            for i, (x, y) in enumerate(training_loader):

                self.__model.reset_gradient()

                x = x.float().to(self.__device)
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

            if not (epoch + 1) % self.__evaluate_every:
                evaluation = self.__evaluator.evaluate_model(data, self.__model)
                metrics["train"] += [evaluation["train"]["metrics"]]
                metrics["val"] += [evaluation["val"]["metrics"]]
                metrics["test"] += [evaluation["test"]["metrics"]]

                if self.__early_stopping_check(metrics["val"][-1][self.__es_metric]):
                    break

                if ((self.__max_lr_decreases > 0 and self.__epochs_no_improvement > 0) and
                        (self.__epochs_no_improvement % self.__lr_decay_patience == 0) and
                        (self.__num_lr_decreases < self.__max_lr_decreases)):
                    self.__decay_learning_rate()

        print("\n Finished train! \n")

        return self.__model, metrics

    def __early_stopping_check(self, metric_value: float) -> bool:
        """
        Decides whether or not to early stop the train based on the early stopping conditions
        @param metric_value: the monitored val metrics (e.g. auc, loss)
        @return: a flag indicating whether or not the training should be early stopped
        """

        if self.__es_metric_trend == "increasing":
            metrics_check = metric_value > self.__es_metric_best_value
        else:
            metrics_check = metric_value < self.__es_metric_best_value

        if metrics_check:
            print("\n\t Old best val {m}: {o:.4f} | New best {m}: {n:.4f} \n"
                  .format(m=self.__es_metric, o=self.__es_metric_best_value, n=metric_value))

            print("\t Saving new best model...")
            self.__model.save(self.__path_to_best_model)
            print("\t -> New best model saved!")

            self.__es_metric_best_value = metric_value
            self.__epochs_no_improvement = 0
        else:
            self.__epochs_no_improvement += 1
            if self.__epochs_no_improvement == self.__patience:
                print("\n ** No decrease in val {} for {} evaluations. Early stopping! ** \n"
                      .format(self.__es_metric, self.__patience, self.__evaluate_every))
                return True

        print("\n Epochs without improvement: ", self.__epochs_no_improvement)
        print("\n ........................................................... \n")
        return False

    def __decay_learning_rate(self):
        """
        Decays the learning rate according to the given decay ratio and reloads the latest best model
        """
        print("\n ** No improvement in val {} in {} epochs. Reducing learning rate. ** \n"
              .format(self.__es_metric, self.__epochs_no_improvement))

        print("\n\t - Old learning rate: ", self.__lr)
        self.__lr *= self.__lr_decay_ratio
        print("\n\t - New learning rate: ", self.__lr)

        self.__num_lr_decreases += 1

        self.__model.load(self.__path_to_best_model)
        self.__model.set_optimizer(self.__optimizer_type, self.__lr)
