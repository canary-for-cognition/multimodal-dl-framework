from classifier.classes.core.Evaluator import Evaluator
from classifier.classes.factories.ModelFactory import ModelFactory
from classifier.classes.utils.Params import Params


class Trainer:

    def __init__(self, train_params: dict, path_to_best_model: str):
        """
        :param train_params: the train related params in the experiment.json file
        :param path_to_best_model: the path at which the best model is saved during train
        """
        self.__path_to_best_model = path_to_best_model

        self.__device = train_params["device"]
        self.__epochs = train_params["epochs"]
        self.__optimizer_type = train_params["optimizer"]
        self.__lr = train_params["learning_rate"]

        self.__log_every = train_params["log_every"]
        self.__evaluate_every = train_params["evaluate_every"]

        self.__patience = train_params["early_stopping"]["patience"]
        self.__es_metric = train_params["early_stopping"]["metrics"]
        self.__es_metric_trend = train_params["early_stopping"]["metrics_trend"]
        self.__es_metric_best_value = 0.0 if self.__es_metric_trend == "increasing" else 1000
        self.__epochs_no_improvement = 0

        network_type, criterion_type = train_params["network_type"], train_params["criterion"]

        network_params = Params.load_network_params(network_type)
        network_params["device"] = self.__device

        self.__model = ModelFactory().get(network_type, network_params)
        self.__model.set_optimizer(self.__optimizer_type, self.__lr)
        self.__model.set_criterion(criterion_type)

        self.__evaluator = Evaluator(self.__device)

    def train(self, data: dict) -> tuple:
        """
        Trains the model according to the established parameters and the given data
        :param data: a dictionary of data loaders containing train, val and test data
        :return: the evaluation metrics of the training and the trained model
        """
        print("\n Training the model... \n")

        self.__model.print_model_overview()

        evaluations = []
        training_loader = data["train"]

        for epoch in range(self.__epochs):
            print("\n *** Epoch {}/{} *** \n".format(epoch + 1, self.__epochs))

            self.__model.train_mode()

            running_loss, running_accuracy = 0.0, 0.0

            for i, (x, y) in enumerate(training_loader):

                self.__model.reset_gradient()

                x, y = x.float().to(self.__device), y.long().to(self.__device)
                o = self.__model.predict(x).to(self.__device)

                running_loss += self.__model.update_weights(o, y)
                running_accuracy += Evaluator.batch_accuracy(o, y)

                if not (i + 1) % self.__log_every:
                    avg_loss, avg_accuracy = running_loss / self.__log_every, running_accuracy / self.__log_every
                    print("[ Epoch: {}/{}, batch: {} ] [ Loss: {:.4f} | Accuracy: {:.4f} ]"
                          .format(epoch + 1, self.__epochs, i + 1, avg_loss, avg_accuracy))
                    running_loss, running_accuracy = 0.0, 0.0

            print("\n ........................................................... \n")

            if not (epoch + 1) % self.__evaluate_every:
                evaluations += [self.__evaluator.evaluate(data, self.__model)]
                if self.__early_stopping_check(evaluations[-1]["metrics"]["val"][self.__es_metric]):
                    break

        print("\n Finished train! \n")

        return self.__model, evaluations

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
