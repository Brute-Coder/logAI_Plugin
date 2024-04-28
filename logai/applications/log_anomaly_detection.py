"""
In this revised version, the anomaly detection algorithm is dynamically loaded based 
on the configuration provided in WorkFlowConfig. The algorithm is retrieved from the 
plugin registry using the specified algorithm name, and then it's instantiated with the 
provided parameters before being used for training and prediction. This approach allows 
for flexibility in choosing different anomaly detection algorithms without modifying the 
LogAnomalyDetection class directly.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from logai.applications.application_interfaces import WorkFlowConfig
from logai.dataloader.data_loader import FileDataLoader
from logai.dataloader.data_model import LogRecordObject
from logai.dataloader.openset_data_loader import OpenSetDataLoader
from logai.information_extraction.categorical_encoder import CategoricalEncoder
from logai.information_extraction.feature_extractor import FeatureExtractor
from logai.information_extraction.log_parser import LogParser
from logai.information_extraction.log_vectorizer import LogVectorizer
from logai.preprocess.preprocessor import Preprocessor
from logai.utils import constants, evaluate
from logai.algorithms.factory import AlgorithmRegistry

class LogAnomalyDetection:
    def __init__(self, config: WorkFlowConfig):
        self.config = config
        self._timestamps = pd.DataFrame()
        self._attributes = pd.DataFrame()
        self._feature_df = pd.DataFrame()
        self._counter_df = pd.DataFrame()
        self._loglines = pd.DataFrame()
        self._log_templates = pd.DataFrame()
        self._ad_results = pd.DataFrame()
        self._labels = pd.DataFrame()
        self._index_group = pd.DataFrame()
        self._loglines_with_anomalies = pd.DataFrame()
        self._group_anomalies = None

    # Properties omitted for brevity

    def execute(self):
        logrecord = self._load_data()
        preprocessed_logrecord = self._preprocess(logrecord)
        loglines = preprocessed_logrecord.body[constants.LOGLINE_NAME]
        parsed_loglines = self._parse(loglines)
        feature_extractor = FeatureExtractor(self.config.feature_extractor_config)
        self._counter_df = feature_extractor.convert_to_counter_vector(
            timestamps=logrecord.timestamp[constants.LOG_TIMESTAMPS],
            attributes=self.attributes,
        )
        algo_config = self.config.anomaly_detection_config
        algo_params = algo_config.params
        algo_name = algo_config.algo_name
        algo_class = AlgorithmRegistry().get_algorithm_instance("anomaly_detection", algo_name)
        anomaly_detector = algo_class(algo_params)
        anomaly_detector.fit(self._counter_df)
        anomalies = anomaly_detector.predict(self._counter_df)["anom_score"]
        self._ad_results = pd.DataFrame(anomalies.rename("result"))

        anomaly_group_indices = self._ad_results[self._ad_results["result"] > 0.0].index.values
        anomaly_indices = []
        for indices in self._index_group["event_index"].iloc[anomaly_group_indices]:
            anomaly_indices += indices

        df = pd.DataFrame(self.loglines)
        df["_id"] = df.index.values
        df["is_anomaly"] = [True if i in anomaly_indices else False for i in df["_id"]]
        self._loglines_with_anomalies = df

    # Methods omitted for brevity

    def _load_data(self):
        if self.config.open_set_data_loader_config is not None:
            dataloader = OpenSetDataLoader(self.config.open_set_data_loader_config)
            logrecord = dataloader.load_data()
        elif self.config.data_loader_config is not None:
            dataloader = FileDataLoader(self.config.data_loader_config)
            logrecord = dataloader.load_data()
        else:
            raise ValueError(
                "data_loader_config or open_set_data_loader_config is needed to load data."
            )
        return logrecord

    # Remaining methods omitted for brevity
