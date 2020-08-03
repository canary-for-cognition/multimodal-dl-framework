import torch

from classifier.classes.core.ExperimentManager import ExperimentManager
from classifier.classes.data.Dataset import Dataset
from classifier.classes.factories.LoaderFactory import LoaderFactory
from classifier.classes.utils.Params import Params


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment_params = Params.load_experiment_params()
    experiment_id = experiment_params["id"]
    device_type = experiment_params["device"]
    dataset_type = experiment_params["dataset_type"]
    network_type = experiment_params["network_type"]

    training_params = experiment_params["training"]
    dataset_params = Params.load_dataset_params(dataset_type)
    cv_metadata = Params.load_cv_metadata(dataset_params["paths"]["cv_metadata"])
    data_loader = LoaderFactory().get(network_type)

    batch_size = training_params["batch_size"]
    base_seed = experiment_params["base_seed"]
    num_seeds = experiment_params["num_seeds"]

    print("\n**********************************************************\n"
          "==========================================================\n"
          "         Neural network for {} prediction               \n"
          "==========================================================\n"
          "**********************************************************\n".format(dataset_type))

    device = ExperimentManager.get_device(device_type)
    print("\n\t\t Using Torch version ... : {v} \n"
          "\n\t\t Running on device ..... : {d} \n".format(d=device, v=torch.__version__))

    ExperimentManager.set_random_seed(base_seed, device)

    dataset = Dataset(dataset_params, batch_size, data_loader, device)

    experiment_manager = ExperimentManager(experiment_id,
                                           network_type,
                                           dataset_type,
                                           dataset,
                                           cv_metadata,
                                           training_params,
                                           device)

    experiment_manager.run_experiment(base_seed, num_seeds)


if __name__ == "__main__":
    main()
