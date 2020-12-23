import torch

from classifier.classes.core.ExperimentManager import ExperimentManager
from classifier.classes.data.Dataset import Dataset
from classifier.classes.factories.LoaderFactory import LoaderFactory
from classifier.classes.utils.Params import Params


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment_params = Params.load_experiment_params()
    dataset_type = experiment_params["dataset_type"]
    network_type = experiment_params["network_type"]
    device_type = experiment_params["device"]
    device = ExperimentManager.get_device(device_type)

    train_params = experiment_params["train"]
    train_params["network_type"] = network_type
    train_params["device"] = device
    dataset_params = Params.load_dataset_params(dataset_type)
    cv_metadata = Params.load_cv_metadata(dataset_params["paths"]["cv_metadata"])
    data_loader = LoaderFactory().get(network_type)

    batch_size = train_params["batch_size"]
    base_seed = experiment_params["base_seed"]
    num_seeds = experiment_params["num_seeds"]

    print("\n**********************************************************\n"
          "==========================================================\n"
          "         Neural network for {} prediction               \n"
          "==========================================================\n"
          "**********************************************************\n".format(dataset_type.upper()))

    print("\n\t\t Using Torch version ... : {v}"
          "\n\t\t Running on device ..... : {d}".format(d=device, v=torch.__version__))

    ExperimentManager.set_random_seed(base_seed, device)

    dataset = Dataset(dataset_params, batch_size, data_loader)

    experiment_manager = ExperimentManager(dataset, cv_metadata, train_params)
    experiment_manager.run_experiment(base_seed, num_seeds)


if __name__ == "__main__":
    main()
