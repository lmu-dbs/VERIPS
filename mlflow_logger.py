import logging
import pathlib
from typing import Dict, Any, Tuple, Sequence

import mlflow
import pandas as pd

from settings import DATA_ROOT
from utils import flatten_dict


class MLFlowLogger:
    def __init__(
        self,
        root: str = DATA_ROOT + '/experiments',
        tracking_uri: str = 'http://usedom.dbs.ifi.lmu.de:5000/',
        experiment_name: str = 'jahnp_VeriPaper',
        artifact_location="/nfs/data3/jahnp/ma/experiments"
    ):
        """
        Constructor.

        Connects to a running MLFlow instance in which the runs of the experiments are stored.
        Also creates an output root directory. The directory will be created if it does not exist.

        :param root: str
            The path of the output root.
        :param tracking_uri: str
            The uri where the MLFlow instance is running.
        :param experiment_name: str
            The name of the experiment on the MLFlow instance server.
        """
        mlflow.set_tracking_uri(uri=tracking_uri)
        experiment = mlflow.get_experiment_by_name(name=experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
        else:
            experiment_id = experiment.experiment_id
        self.experiment_id = experiment_id
        self.root = pathlib.Path(root)
        self.current_run = None

    def init_experiment(
        self,
        name,
        hyper_parameters: Dict[str, Any],
    ) -> Tuple[str, pathlib.Path]:
        """
        Initialise an experiment, i.e. a run in the specified experiment.
        Creates an entry describing the run's hyper-parameters, and associates an unique output directory.

        :param hyper_parameters: Dict
            The hyperparameters. Should be serialisable to JSON/BSON.
        :return: A tuple (id, path) where
            id:
                a unique ID of the experiment (equal to the ID in the MLFlow instance)
            path:
                the path to an existing directory for outputs of the experiment.
        """
        self.current_run = mlflow.start_run(run_name=name, experiment_id=self.experiment_id)

        # create output directory
        output_path = self.root / str(self.current_run.info.run_id)
        if output_path.is_dir():
            logging.error('Output path already exists! {p}'.format(p=output_path))
        output_path.mkdir(exist_ok=True, parents=True)
        hyper_parameters["out_dir"] = output_path

        self.log_params(hyper_parameters=flatten_dict(hyper_parameters))
        # return id as experiment handle, and path of existing directory to store outputs to
        return self.current_run.info.run_id, output_path

    @staticmethod
    def log_params(hyper_parameters: Dict[str, Any]):
        mlflow.log_params(params=flatten_dict(hyper_parameters))

    @staticmethod
    def finalise_experiment() -> None:
        """
        Close the current run.
        :return: None.
        """
        mlflow.end_run()

    @staticmethod
    def log_results(result: Dict[str, Any], step=None, file=None, curRunStore = None):
        """
        :param file: Path to logging file
        :param result: Dict
            A flattened dictionary holding high-level results, e.g. a few numbers.
        :param step: int
        :return: None.
        """
        mlflow.log_metrics(metrics=flatten_dict(result), step=step)
        mlflow.log_artifact(local_path = file)
        mlflow.log_artifact(local_path = curRunStore)


    def get_entries(
        self,
        keys: Sequence[str],
        equals: Sequence[bool],
        values: Sequence[str]
    ) -> pd.DataFrame:
        """
        Get entries from Mlflow client in form of pandas dataframe.
        :param keys: keys to filter by.
        :param equals: whether key should be equal or not equal to value.
        :param values: values to filter by. The order must match <keys>
        :return: pd.DataFrame containing the result
        """
        search_string = ''
        for index, key in enumerate(keys):
            operand = '=' if equals[index] else '!='
            tmp_str = f'{key} {operand} "{values[index]}"'
            if index != 0:
                tmp_str = f' and {tmp_str}'
            search_string = search_string + tmp_str
        return mlflow.search_runs(experiment_ids=self.experiment_id, filter_string=search_string)
