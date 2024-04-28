"""
anomaly_detector_het.py compatible with plugin_discovery_algorithm_factory.py as a plugin, we need to ensure that it follows 
the conventions and requirements for plugin registration and interface implementation. Here's how you can modify it:

Registration: Use the factory.register decorator or registration function provided by plugin_discovery_algorithm_factory.py 
to register the anomaly detector plugin.
Interface Implementation: Ensure that HetAnomalyDetector class implements the required methods defined by the AnomalyDetectionAlgo 
interface.
Naming Convention: Name the file and class according to the conventions expected by plugin_discovery_algorithm_factory.py.
"""

# anomaly_detector_het.py

from logai.algorithms.algo_interfaces import AnomalyDetectionAlgo
from logai.analysis.anomaly_detector import AnomalyDetector
from logai.algorithms.factory import factory
from logai.utils import constants
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class HetAnomalyDetectionConfig:
    """
    Heterogeneous Anomaly Detector Parameters.

    :param train_test_ratio: The ratio between test and training splits.
    """

    train_test_ratio: float = 0.3


@factory.registry.register_algorithm("anomaly_detection", "het", HetAnomalyDetectionConfig)
class HetAnomalyDetector(AnomalyDetectionAlgo):
    """
    Anomaly Detector Wrapper to handle heterogeneous log feature dataframe which include various attributes of log. For
    each attribute, we build its specific anomaly detector if the data satisfies the requirement.
    This current version only supports anomaly detection on the constants.LOGLINE_COUNTS field (i.e. frequency count of
    the log events).
    """

    def __init__(self, config: HetAnomalyDetectionConfig):
        self.anomaly_detector_config = config
        self.train_test_ratio = config.train_test_ratio
        self.model_dict = {}

    def preprocess(self, counter_df: pd.DataFrame):
        """
        Splits raw log feature dataframe by unique attribute ID.

        :param counter_df: A log feature dataframe that must contain at least two columns
            ['timestamp': datetime, constants.LOGLINE_COUNTS: int].
            The rest of columns combinations are treated as log attribute ID.
        :return: The processed log feature dataframe.
        """
        ts_df = counter_df[[constants.LOG_COUNTS]]
        ts_df.index = counter_df[constants.LOG_TIMESTAMPS]
        counter_df["attribute"] = counter_df.drop(
            [constants.LOG_COUNTS, constants.LOG_TIMESTAMPS], axis=1
        ).apply(lambda x: "-".join(x.astype(str)), axis=1)
        attr_list = counter_df["attribute"].unique()
        return attr_list

    def fit_predict(self, log_feature: pd.DataFrame) -> pd.DataFrame:
        """
        Trains a model and predicts anomaly scores.

        :param log_features: A log feature dataframe that must contain at least two columns
            ['timestamp': datetime, constants.LOGLINE_COUNTS: int].
            The rest of columns combinations are treated as log attribute ID.
        :return: The predicted anomaly scores.
        """
        res = pd.DataFrame()
        attr_list = self.preprocess(log_feature)

        counter_df = log_feature
        for attr in attr_list:
            attr_df = counter_df[counter_df["attribute"] == attr]

            # if the length of sequence is short, skip the anomaly detection
            if attr_df.shape[0] < constants.MIN_TS_LENGTH:
                anom_score = np.repeat(0.0, attr_df.shape[0])
                trainval = np.repeat(None, attr_df.shape[0])
                timestamps = attr_df[[constants.LOG_TIMESTAMPS]].values.squeeze()
                tmp_dic = {
                    constants.LOG_TIMESTAMPS: timestamps,
                    "anom_score": anom_score,
                    "trainval": trainval,
                }
                tmp_df = pd.DataFrame(tmp_dic, index=attr_df.index)
                res = pd.concat([res, tmp_df])
            else:
                train, test = train_test_split(
                    attr_df[[constants.LOG_TIMESTAMPS, constants.LOG_COUNTS]],
                    shuffle=False,
                    train_size=self.train_test_ratio,
                )
                model = AnomalyDetector(self.anomaly_detector_config)
                train_scores = model.fit(train)
                test_scores = model.predict(test)
                tmp_df = pd.concat([train_scores, test_scores])
                res = pd.concat([res, tmp_df])
                self.model_dict[attr] = model
        assert (
            res.index == counter_df.index
        ).all(), "Res.index should be identical to counter_df.index"
        assert len(res) == len(
            counter_df.index
        ), "length of res should be equal to length of counter_df"
        return res
