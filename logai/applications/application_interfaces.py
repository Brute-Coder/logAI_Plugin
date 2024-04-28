from attr import dataclass

from logai.config_interfaces import Config
from logai.dataloader.data_loader import DataLoaderConfig , FileDataLoader
from logai.dataloader.openset_data_loader import OpenSetDataLoaderConfig , OpenSetDataLoader
from logai.preprocess.partitioner import PartitionerConfig
from logai.preprocess.openset_partitioner import OpenSetPartitionerConfig
from logai.preprocess.preprocessor import PreprocessorConfig , Preprocessor
from logai.information_extraction.log_parser import LogParserConfig , LogParser
from logai.information_extraction.log_vectorizer import VectorizerConfig
from logai.information_extraction.feature_extractor import FeatureExtractorConfig , FeatureExtractor
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig
from logai.analysis.anomaly_detector import AnomalyDetectionConfig
from logai.analysis.nn_anomaly_detector import NNAnomalyDetectionConfig
from logai.analysis.clustering import ClusteringConfig
from logai.dataloader.data_model import LogRecordObject
from logai.utils import constants, evaluate

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# from logai.information_extraction.log_vectorizer import LogVectorizer
# from logai.preprocess.partitioner import PartitionerConfig, Partitioner
# from logai.preprocess.preprocessor import Preprocessor

@dataclass
class WorkFlowConfig(Config):
    """
    Configuration class for the end-to-end workflow.
    """

    data_loader_config: DataLoaderConfig = None
    open_set_data_loader_config: OpenSetDataLoaderConfig = None
    preprocessor_config: PreprocessorConfig = None
    log_parser_config: LogParserConfig = None
    log_vectorizer_config: VectorizerConfig = None
    partitioner_config: PartitionerConfig = None
    open_set_partitioner_config: OpenSetPartitionerConfig = None
    categorical_encoder_config: CategoricalEncoderConfig = None
    feature_extractor_config: FeatureExtractorConfig = None
    anomaly_detection_config: AnomalyDetectionConfig = None
    nn_anomaly_detection_config: NNAnomalyDetectionConfig = None
    clustering_config: ClusteringConfig = None

    @classmethod
    def from_dict(cls, config_dict):
        config = super().from_dict(config_dict)

        for attr_name in config_dict:
            attr_value = getattr(config, attr_name)
            if attr_value is not None:
                # Convert each attribute to its respective configuration class
                config_class = globals()[attr_value.__class__.__name__]
                setattr(config, attr_name, config_class.from_dict(attr_value.as_dict()))

        return config






class LogAnomalyDetection:
    """This is a workflow for log anomaly detection."""
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

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def loglines(self):
        return self._loglines

    @property
    def log_templates(self):
        return self._log_templates

    @property
    def attributes(self):
        return self._attributes

    @property
    def results(self):
        res = (
            self._loglines_with_anomalies.join(self.attributes)
            .join(self.timestamps)
            .join(self.event_group)
        )
        return res

    @property
    def anomaly_results(self):
        return self.results[self.results["is_anomaly"]]

    @property
    def anomaly_labels(self):
        return self._labels

    @anomaly_labels.setter
    def anomaly_labels(self, labels):
        self._labels = labels

    @property
    def event_group(self):
        event_index_map = dict()
        for group_id, indices in self._index_group["event_index"].items():
            for i in indices:
                event_index_map[i] = group_id

        event_index = pd.Series(event_index_map).rename("group_id")
        return event_index

    @property
    def feature_df(self):
        return self._feature_df

    @property
    def counter_df(self):
        return self._counter_df

    def evaluation(self):
        if self.anomaly_labels is None:
            raise TypeError

        labels = self.anomaly_labels.to_numpy()
        pred = np.array([1 if r else 0 for r in self.results["is_anomaly"]])
        return evaluate.get_accuracy_precision_recall(labels, pred)

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
        if self.config.anomaly_detection_config.algo_name in constants.COUNTER_AD_ALGO:
            # Handle counter-based anomaly detection
            pass
        else:
            # Handle other anomaly detection methods
            pass
        # Continue with execution...

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

    def _preprocess(self, log_record):
        logline = log_record.body[constants.LOGLINE_NAME]

        self._loglines = logline
        self._timestamps = log_record.timestamp
        self._attributes = log_record.attributes.astype(str)

        preprocessor = Preprocessor(self.config.preprocessor_config)
        preprocessed_loglines, _ = preprocessor.clean_log(logline)

        new_log_record = LogRecordObject(
            body=pd.DataFrame(preprocessed_loglines, columns=[constants.LOGLINE_NAME]),
            timestamp=log_record.timestamp,
            attributes=log_record.attributes,
        )

        return new_log_record

    def _parse(self, loglines):
        parser = LogParser(self.config.log_parser_config)
        parsed_results = parser.parse(loglines.dropna())
        parsed_loglines = parsed_results[constants.PARSED_LOGLINE_NAME]
        return parsed_loglines
