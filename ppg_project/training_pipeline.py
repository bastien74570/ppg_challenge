"""Module Training xgboost pipeline for PPG project."""
from typing import Dict, List, Tuple
import itertools

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy.signal import find_peaks
from tqdm import tqdm
import xgboost as xgb

from ppg_project.dtypes import XGBoostTrainingPipelineConfig
from ppg_project import utils


class XGBoostTrainingPipeline:
    """Training pipeline for the PPG project."""

    def __init__(self, config: XGBoostTrainingPipelineConfig):
        """Initializes the training pipeline.

        Args:
        config (XGBoostTrainingPipelineConfig): dict containing the configuration for the training pipeline.
        """
        self.config = config
        self.flatten_feature = True
        self.load_data()
        self.ppg_features_col = [c for c in self.train.columns if c.startswith("ppg")]
        self.crafted_features_col = [
            c for c in self.train.columns if c.startswith("feature")
        ]

    def load_data(self):
        """Loads the data from the data path."""
        print("Loading data...")
        self.train = pd.read_csv(f"{self.config.data_path}/train.csv")
        self.train_label = pd.read_csv(f"{self.config.data_path}/train_labels.csv")
        self.test = pd.read_csv(f"{self.config.data_path}/test.csv")

    def select_model(self):
        """Selects the model to use for the training."""
        self.model = xgb.XGBRegressor(
            max_depth=self.config.max_depth,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
        )

    def run(self):
        """Runs the training pipeline."""
        if self.config.apply_preprocessing:
            print("Applies filtering to the ppg signal.")
            self.train = self.preprocessing_ppg(self.train)

        if self.config.appply_segmentation:
            self.train,self.train_label = self.apply_window_segmentation(self.train, self.train_label)    
        
        self.train = self.extract_features_from_dataframe(self.train)
        if self.config.validation_strategy == "kfold":
            print("Evaluate model on training set with kfold cross validation...")
            results = self.run_kfold(self.train, self.train_label)
        elif self.config.validation_strategy == "prediction_on_test":
            print("Train model and get predictions...")
            results = self.run_on_test(self.train,self.train_label,self.test)
        return results

    def run_kfold(self, data: pd.DataFrame, target: pd.DataFrame, disable:bool = False) -> pd.DataFrame:
        """Run kfold cross validation.

        Args:
            data (pd.DataFrame): dataframe containing the features.
            target (pd.DataFrame): dataframe containing the target.
            disable(bool): disable tqdm progress bar. Default is False.

        Returns:
            pd.DataFrame: dataframe containing the performance metrics.
        """

        kf = model_selection.KFold(
            n_splits=self.config.num_folds, shuffle=True, random_state=42
        )
        score_fold_list = list()
        for i, (train_index, test_index) in tqdm(enumerate(kf.split(data)), total=self.config.num_folds, disable= disable):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            self.select_model()
            
            # Standardize the data (important for PCA)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            if self.config.apply_pca: 
                # Apply PCA to training data
                pca = PCA(n_components=0.95)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
            
            self.training_loop(X_train, y_train)
            predictions = self.get_predictions(X_test)
            score = self.evaluation(predictions, y_test)
            score.update({"fold": i})
            score_fold_list.append(score)
        results = pd.DataFrame(score_fold_list)
        return results

    def run_on_test(self, train_data: pd.DataFrame, train_target: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """Train a model on the training set and get predictions on the test set.

        Args:
            train_data (pd.DataFrame): train dataset to train the model.
            train_target (pd.DataFrame): train target to train the model.
            test_data (pd.DataFrame): test dataset to get predictions.

        Returns:
            pd.DataFrame: predictions on the test set.
        """
        self.select_model()
        if self.config.apply_preprocessing:
            print("Applies filtering to the ppg signal on the test set.") 
            test_data = self.preprocessing_ppg(test_data)
        test_data= self.extract_features_from_dataframe(test_data)
        
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        
        if self.config.apply_pca: 
            pca = PCA(n_components=0.95)
            train_data = pca.fit_transform(train_data)
            test_data = pca.transform(test_data)
        
        self.training_loop(train_data, train_target)
        predictions = self.get_predictions(test_data)
        results = pd.DataFrame(predictions, columns=["predictions"])
        return results
        
    def hyperparameter_tuning(
        self,
        hyperparam_ranges: Dict[str, List[float]],
    ):
        """Run KFOLD validation for a number of hyperpararm configs.

        Args:
            hyperparam_ranges (Dict[str, List[float]]): Dict of hyperparameter values to tune.
        """
        keys, values = zip(*hyperparam_ranges.items())
        param_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        all_results = []
        best_score = 0
        best_config = self.config
        df = self.extract_features_from_dataframe(self.train)
        for params in tqdm(param_permutations):
            self.config = self.create_new_config(params)
            self.model = self.select_model()
            result = self.run_kfold(df, self.train_label,disable=True)
            result["params"] = str(params)
            score = result["r2"].mean()
            if score > best_score:
                best_score = score
                best_config = self.config
            all_results.append(result)
        all_results = pd.concat(all_results)
        print(f"Best Config: {best_config.__dict__}")
        print(f"Best Performance: {best_score}")
        return all_results, best_config

    def apply_window_segmentation(self, data: pd.DataFrame, target: pd.DataFrame) -> Tuple[np.array, np.array]:
        print(f"Segment data in window of {self.config.segment_window_sec} seconds.")
        data_window = utils.segment_data_in_window(
                data, target, nsec=self.config.segment_window_sec, sampling_rate=self.config.sampling_rate
            )
        self.ppg_features_col = [c for c in data_window.columns if c.startswith("ppg")]
        return data_window.drop(columns=["target"]), data_window["target"]
    
    def create_new_config(self, params: Dict[str, float]) -> XGBoostTrainingPipelineConfig:
        """Insert given hyperparameters into a new pipeline config.

        Args:
            params (Dict[str, float]): parameters to insert.

        Returns:
            XGBoostTrainingPipelineConfig: new config.
        """
        return XGBoostTrainingPipelineConfig(
            data_path=self.config.data_path,
            **params,
        )

    def training_loop(self, data: pd.DataFrame, target: pd.DataFrame):
        """Run training loop.

        Args:
            data (pd.DataFrame): dataframe containing the features.
            target (pd.DataFrame): dataframe containing the target.
        """
        (
            data_train,
            data_test,
            labels_train,
            labels_test,
        ) = model_selection.train_test_split(
            data, target, test_size=0.2, random_state=42
        )
        self.model.fit(
            data_train,
            labels_train,
            eval_set=[
                (data_train, labels_train),
                (data_test, labels_test),
            ],
            verbose=False,
        )

    def get_predictions(self, data: pd.DataFrame) -> np.array:
        """Returns the predictions for the data.

        Args:
            data (pd.DataFrame): data to predict.
        Returns:
            np.array: predictions.
        """
        predictions = self.model.predict(data)
        return predictions

    def evaluation(self, y_pred: np.array, y_test: np.array) -> Dict[str, float]:
        """Evaluate the performance of the model.

        Args:
            y_pred (np.array): predictions of the model.
            y_test (np.array): ground truth values.

        Returns:
            Dict[str, float]: dictionary containing the performance metrics and feature performances.
        """
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmsle = np.log(rmse)
        r2 = metrics.r2_score(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        feature_importances = dict(zip(self.features, self.model.feature_importances_))
        return {
            "rmse": rmse,
            "mae": mae,
            "rmsle": rmsle,
            "r2": r2,
            "feature_importances": feature_importances,
        }

    def preprocessing_ppg(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing ppg signals with bandpass butterworth filter.

        Args:
            data (pd.DataFrame): dataframe containing ppg values.

        Returns:
            pd.DataFrame: dataframe containing ppg values after filtering.
        """
        data_ppg_filtered= data.copy()
        data_ppg = data[self.ppg_features_col].values
        for i in range(data_ppg.shape[0]):
            data_ppg[i, :] = utils.butter_bandpass_filter(
                data_ppg[i, :],
                lowcut=self.config.lowcut,
                highcut=self.config.highcut,
                sampling_rate=self.config.sampling_rate,
                order=3,
            )
        data_ppg_filtered[self.ppg_features_col] = data_ppg
        return data_ppg_filtered

    def extract_features_from_dataframe(self,data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from dataframe.

        Args:
            data (pd.DataFrame): original dataframe.

        Returns:
            pd.DataFrame: transformed features.
        """
        list_df_feature = list()
        for feature in self.config.features_list:
            print("Feature extraction: ", feature)
            df_feature = getattr(self,f"get_{feature}_features")(data)
            list_df_feature.append(df_feature)
        df = pd.concat(list_df_feature, axis=1)
        self.features = df.columns
        return df  

    def get_spectrogram_features(self, data: np.array) -> pd.DataFrame:
        """Compute spectrogram features.

        Args:
            data (np.array): ppg data to compute spectrogram features.

        Returns:
            pd.DataFrame: dataframe with spectrogram features.
        """
        data = data[self.ppg_features_col].values
        df_list = list()
        for i in tqdm(range(data.shape[0])):
            freq, _, feature = utils.get_log_spectrogram_features(
                data[i, :], sampling_rate=self.config.sampling_rate
            )
            if i == 0:
                index = np.where(freq <= self.config.highcut)[0]
            spec_mean = np.mean(feature[index, :],axis=1)
            spec_std = np.std(feature[index, :],axis=1)
            feature = np.concatenate([spec_mean,spec_std])
            df = pd.DataFrame(feature)
            df_list.append(df.T)
        list_mean_spec_col = [f"spectrogram_mean_{i}" for i in range(spec_mean.shape[0])]
        list_std_spec_col = [f"spectrogram_std_{i}" for i in range(spec_std.shape[0])]
        df_spec = pd.concat(df_list, axis=0).reset_index(drop=True)
        df_spec.columns = list_mean_spec_col + list_std_spec_col
        return df_spec

    def get_wavelet_features(self, data: np.array) -> pd.DataFrame:
        """Compute wavelet features.

        Args:
            data (np.array): ppg data to compute wavelet features.

        Returns:
            pd.DataFrame: dataframe with spectrogram features.
        """
        data = data[self.ppg_features_col].values
        data = data[:,::30] #this is only to reduce the computation power needed for computing wavelet features. 
        df_list = list()
        for i in tqdm(range(data.shape[0])):
            freq, feature = utils.get_wavelet_features(
                data[i, :], sampling_rate=self.config.sampling_rate
            )
            if i == 0:
                index = np.where(freq <= self.config.highcut)[0]
            wavelet_mean = np.mean(feature[index, :],axis=1)
            wavelet_std = np.std(feature[index, :],axis=1)
            feature = np.concatenate([wavelet_mean,wavelet_std])
            df = pd.DataFrame(feature)
            df_list.append(df.T)
        list_mean_wav_col = [f"wavelet_mean_{i}" for i in range(wavelet_mean.shape[0])]
        list_std_wav_col = [f"wavelet_std_{i}" for i in range(wavelet_std.shape[0])]
        df_wav = pd.concat(df_list, axis=0).reset_index(drop=True)
        df_wav.columns = list_mean_wav_col + list_std_wav_col
        return df_wav
    
    def get_peak_features(self, data: np.array) -> pd.DataFrame:
        """Compute peak features.

        Args:
            data (np.array): ppg data to compute peak features.

        Returns:
            pd.DataFrame: dataframe with peak features.
        """
        data = data[self.ppg_features_col].values
        list_dict = list()
        for i in tqdm(range(data.shape[0])):
            peaks, properties = find_peaks(data[i, :], prominence=0, width=0)

            mean_diff_peak, std_diff_peak = np.mean(np.diff(peaks)), np.std(
                np.diff(peaks)
            )
            mean_prominences, std_prominences = np.mean(
                properties["prominences"]
            ), np.std(properties["prominences"])
            mean_widths, std_widths = np.mean(properties["widths"]), np.std(
                properties["widths"]
            )
            list_dict.append(
                {
                    "peak": len(peaks),
                    "peak_mean_diff": mean_diff_peak,
                    "peak_mean_prominences": mean_prominences,
                    "peak_mean_widths": mean_widths,
                    "peak_std_diff": std_diff_peak,
                    "peak_std_prominences": std_prominences,
                    "peak_std_widths": std_widths,
                }
            )
        return pd.DataFrame(list_dict)
    
    def get_features_features(self, data:pd.DataFrame) -> pd.DataFrame:
        """Get features extracted from dataframe.

        Args:
            data (pd.DataFrame): dataframe with all features.

        Returns:
            pd.DataFrame: dataframe with crafted features.
        """
        return data[self.crafted_features_col]
    
    def get_ppg_features(self, data:pd.DataFrame) -> pd.DataFrame:
        """Compute ppg features over the dataframe.

        Args:
            data (pd.DataFrame): dataframe with all features.

        Returns:
            pd.DataFrame: dataframe with transformed ppg features.
        """
        data["ppg_mean"] = data[self.ppg_features_col].mean(axis=1)
        data["ppg_std"] = data[self.ppg_features_col].std(axis=1)
        return data[["ppg_mean","ppg_std"]]
