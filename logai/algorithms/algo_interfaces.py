"""
This modified version ensures that the interfaces are properly defined and can be
imported and used by the plugin classes. Ensure that the plugin classes implement 
these interfaces correctly, providing concrete implementations for all the abstract 
methods defined in the interfaces. This will make the plugin classes compatible with
the factory and registry mechanism in plugin_discovery_algorithm_factory.py.
"""

import abc
import pandas as pd

class ParsingAlgo(abc.ABC):
    """
    Interface for parsing algorithms.
    """

    @abc.abstractmethod
    def fit(self, loglines: pd.Series):
        """
        Fit parsing algorithm with input.

        :param loglines: pd.Series of loglines as input
        """
        pass

    @abc.abstractmethod
    def parse(self, loglines: pd.Series) -> pd.DataFrame:
        """
        Parse loglines.

        :param loglines: pd.Series of loglines to parse
        :return: pd.DataFrame of parsed results ["loglines", "parsed_loglines", "parameter_list"].
        """
        pass


class VectorizationAlgo(abc.ABC):
    """
    Interface for logline vectorization algorithms.
    """

    @abc.abstractmethod
    def fit(self, loglines: pd.Series):
        """
        Fit vectorizer with input.

        :param loglines: pd.Series of loglines as input.
        """
        pass

    @abc.abstractmethod
    def transform(self, loglines: pd.Series) -> pd.DataFrame:
        """
        Transform given loglines into vectors.

        :param loglines: pd.Series of loglines to transform.
        :return: pd.DataFrame of vectorized loglines.
        """
        pass


class ClusteringAlgo(abc.ABC):
    """
    Interface for clustering algorithms.
    """

    @abc.abstractmethod
    def fit(self, log_features: pd.DataFrame):
        """
        Fit clustering algorithm with input.

        :param log_features: pd.DataFrame of log features.
        """
        pass

    @abc.abstractmethod
    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """
        Predict using trained clustering model.

        :param log_features: pd.DataFrame of log features.
        :return: pd.Series of cluster labels.
        """
        pass


class AnomalyDetectionAlgo(abc.ABC):
    """
    Interface for anomaly detection algorithms.
    """

    @abc.abstractmethod
    def fit(self, log_features: pd.DataFrame):
        """
        Fit anomaly detection algorithm with input.

        :param log_features: pd.DataFrame of log features.
        """
        pass

    @abc.abstractmethod
    def predict(self, log_features: pd.DataFrame) -> pd.Series:
        """
        Predict using trained anomaly detection model.

        :param log_features: pd.DataFrame of log features.
        :return: pd.Series of anomaly scores.
        """
        pass


class CategoricalEncodingAlgo(abc.ABC):
    """
    Interface for categorical encoders.
    """

    @abc.abstractmethod
    def fit_transform(self, log_attributes: pd.DataFrame) -> pd.DataFrame:
        """
        Fit encoder and transform categorical attributes into numerical representations.

        :param log_attributes: pd.DataFrame of log attributes.
        :return: pd.DataFrame of encoded log attributes.
        """
        pass
