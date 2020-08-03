import torch

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
        self.__early_stopping = training_params["early_stopping"]["active"]
        self.__early_stopping_patience = training_params["early_stopping"]["patience"]
        self.__monitored_metrics = training_params["early_stopping"]["metrics"]
        self.__monitored_metrics_trend = training_params["early_stopping"]["metrics_trend"]
        self.__monitored_metrics_best_value = 0.0 if self.__monitored_metrics_trend == "increasing" else 1000
        self.__epochs_without_improvement = 0

        self.__learning_rate = training_params["learning_rate"]["initial_value"]
        self.__learning_rate_decay_ratio = training_params["learning_rate"]["decay_ratio"]
        self.__learning_rate_decay_patience = training_params["learning_rate"]["decay_patience"]
        self.__max_learning_rate_decreases = training_params["learning_rate"]["max_decreases"]
        self.__reload_best_on_decay = training_params["learning_rate"]["reload_best_on_decay"]
        self.__num_learning_rate_decreases = 0

        self.__use_tbptt = training_params["tbptt"]["active"]
        self.__k1 = training_params["tbptt"]["k1"]
        self.__k2 = training_params["tbptt"]["k2"]

        self.__model = self.__create_model()
        self.__evaluator = Evaluator(self.__batch_size, self.__device)

    def get(self) -> Model:
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

        epochs_metrics = {"training": [], "validation": [], "test": []}

        for epoch in range(self.__epochs):
            print("\n *** Epoch {} *** ".format(epoch + 1))

            self.__model.train_mode()
            self.__train_epoch(data["training"], epoch)

            evaluation = self.__evaluator.evaluate_model(data, self.__model)
            epochs_metrics["training"] += [evaluation["training"]["metrics"]]
            epochs_metrics["validation"] += [evaluation["validation"]["metrics"]]
            epochs_metrics["test"] += [evaluation["test"]["metrics"]]

            if self.__early_stopping:
                self.__update_early_stopping_metrics(epochs_metrics["validation"][-1][self.__monitored_metrics])
                if self.__epochs_without_improvement == self.__early_stopping_patience:
                    print("\n ** No decrease in validation loss in {} epochs. Stopping training early ** \n"
                          .format(self.__early_stopping_patience))
                    break

            if ((self.__max_learning_rate_decreases > 0 and self.__epochs_without_improvement > 0) and
                    (self.__epochs_without_improvement % self.__learning_rate_decay_patience == 0) and
                    (self.__num_learning_rate_decreases < self.__max_learning_rate_decreases)):
                self.__decay_learning_rate()

        print("\n Finished training! \n")

        return epochs_metrics

    def __tbptt(self, inputs: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Performs TBPTT (Truncated Backpropagation Through Time)
        Reference: https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/4
        :param inputs: a batch of input sequences of shape [batch_size, sequence_length, num_features]
        :param labels: a batch of labels for the inputs
        :return: the loss and accuracy for the processed batch of inputs
        """
        batch_size, sequence_length = inputs.shape[0], inputs.shape[1]
        batch_loss, batch_accuracy = 0, 0
        retain_graph = self.__k1 < self.__k2

        for i in range(batch_size):
            print("i: ", i)

            item_loss, item_accuracy = 0, 0

            states = [(None, self.__model.get_init_state(1))]
            inputs, label = inputs[i], torch.tensor([labels[i].item()])

            for j in range(sequence_length):
                print("j: ", j)

                state = states[-1][1].detach()
                state.requires_grad = True
                state.retain_grad()
                time_step = inputs[j, :].unsqueeze(0).unsqueeze(1)

                output, new_state = self.__model.predict(time_step, state)
                output, new_state = output.to(self.__device), new_state.to(self.__device)
                new_state.retain_grad()
                states.append((state.clone(), new_state.clone()))

                # Delete old states
                while len(states) > self.__k2:
                    del states[0]

                # Backprop every k1 steps
                if (j + 1) % self.__k1 == 0:
                    print("\n k1 = {} triggered \n".format(self.__k1))

                    item_loss += self.__model.update_weights(output, label, retain_graph, optimize=False)
                    item_accuracy += Evaluator.compute_batch_accuracy(output, label)

                    for z, s in enumerate(states):
                        print("State index:", z)
                        print("s0:", s[0].requires_grad, s[0].grad, s[0].is_leaf)
                        print("s1:", s[1].requires_grad, s[1].grad, s[1].is_leaf)

                    # Backprop over the last k2 states
                    for k in range(self.__k2 - 1):
                        print("\n k2 = {} triggered \n".format(self.__k2))
                        if states[-k - 2][0] is None:
                            break

                        print(states[-k - 1][0])
                        curr_gradient = states[-k - 1][0].grad
                        print("curr_gradient:", curr_gradient)
                        states[-k - 2][1].backward(curr_gradient, retain_graph=True)

                    self.__model.optimize()

                batch_loss += item_loss
                batch_accuracy += item_accuracy

        return batch_loss, batch_accuracy

    def __train_epoch(self, training_loader: torch.utils.data.DataLoader, epoch: int):
        """
        Trains the model batch-wise for the current epoch
        :param training_loader: the data which to train the model on
        :param epoch: the number identifying the current epoch
        """
        running_accuracy, running_loss = 0.0, 0.0

        for i, (inputs, labels) in enumerate(training_loader):

            labels = labels.long().to(self.__device)

            if self.__use_tbptt:
                batch_loss, batch_accuracy = self.__tbptt(inputs, labels)
            else:
                outputs = self.__model.predict(inputs).to(self.__device)
                batch_loss = self.__model.update_weights(outputs, labels)
                batch_accuracy = Evaluator.compute_batch_accuracy(logit=outputs, target=labels)

            running_loss += batch_loss
            running_accuracy += batch_accuracy

            if (i + 1) % self.__log_every == 0:
                print("\n [ Epoch: {}, batches: {} ] loss: {:.5f} | accuracy: {:.5f}"
                      .format(epoch + 1, i + 1, running_loss / self.__log_every, running_accuracy / self.__log_every))
                running_loss, running_accuracy = 0.0, 0.0

        print("\n ........................................................... \n")

    def __update_early_stopping_metrics(self, monitored_metrics: float):
        """
        Decides whether or not to early stop the training based on the early stopping conditions
        :param monitored_metrics: the monitored validation metrics (e.g. AUC, loss)
        """
        if self.__monitored_metrics_trend == "increasing":
            metrics_check = monitored_metrics > self.__monitored_metrics_best_value
        else:
            metrics_check = monitored_metrics < self.__monitored_metrics_best_value

        if metrics_check:
            print("\n\t Old best {m}: {old} | New best {m}: {new} \n".format(m="validation " + self.__monitored_metrics,
                                                                             old=self.__monitored_metrics_best_value,
                                                                             new=monitored_metrics))
            print("\t New best model found! Saving now...")
            self.__model.save(self.__path_to_best_model)
            self.__monitored_metrics_best_value = monitored_metrics
            self.__epochs_without_improvement = 0
        else:
            self.__epochs_without_improvement += 1

        print("\n Epochs without improvement: ", self.__epochs_without_improvement)
        print("\n ........................................................... \n")

    def __decay_learning_rate(self):
        """
        Decays the learning rate according to the given decay ratio and reloads the latest best model
        """
        print("\n ** No improvement in validation {} score in {} epochs. Reducing learning rate. ** \n"
              .format(self.__monitored_metrics, self.__epochs_without_improvement))

        print("\n\t - Old learning rate: ", self.__learning_rate)
        self.__learning_rate *= self.__learning_rate_decay_ratio
        print("\n\t - New learning rate: ", self.__learning_rate)

        self.__num_learning_rate_decreases += 1

        if self.__reload_best_on_decay:
            self.__model.load(self.__path_to_best_model)

        self.__model.set_optimizer(self.__optimizer_type, self.__learning_rate, self.__clip_gradient)
