"""Module training of deep learning model for PPG project."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppg_project.dtypes import DLTrainingPipelineConfig
from ppg_project.ppg_dataloader import PPGDataloader
from ppg_project import utils


class DLTrainingPipeline:
    def __init__(
        self,
        config: DLTrainingPipelineConfig,
        model: torch.nn.Module,
    ):
        """Initialise Training pipeline for DL model.

        Args:
            config (DLPipelineConfig): configuration details for pipeline
            model (torch.nn.Module): model to train
        """
        self.config = config
        self.model = model
        self.load_data()
        
    def load_data(self):
        """Loads the data from the data path."""
        print("Loading data...")
        self.train = pd.read_csv(f"{self.config.data_path}/train.csv")
        self.train_label = pd.read_csv(f"{self.config.data_path}/train_labels.csv")
        self.test = pd.read_csv(f"{self.config.data_path}/test.csv")
        
    def process_data(self, data: np.array, target: pd.DataFrame) -> pd.DataFrame:
        """Process data for training.

        Args:
            data (np.array): data array containing ppg values.
            target (pd.DataFrame): target dataframe containing labels.

        Returns:
            pd.DataFrame: dataframe containing processed data.
        """
        df_list = list()
        for i in tqdm(range(data.shape[0])):
            data[i,:] = utils.butter_bandpass_filter(data[i,:],0.5,10,100)
            if self.config.dl_features == "spectrogram":
                frequencies, _, feature = utils.get_log_spectrogram_features(data[i,:],100)
                index = np.where(frequencies<=5)[0]
                feature = feature[index,:]
            elif self.config.dl_features == "wavelet":
                _,feature = utils.get_wavelet_features(data[i,:],100,scales=np.arange(1,128))
            if i == 0:
                print(feature.shape)
            df = pd.DataFrame({"dl_feature":[feature]})
            df_list.append(df)
        df_spec = pd.concat(df_list,axis=0).reset_index(drop=True)
        data = pd.concat([df_spec,target],axis=1)
        return data

    def run_holdhout(self, data: pd.DataFrame) -> pd.DataFrame:
        """Train and evaluate a model using train-test split on sample level.

        Args:
            data (pd.DataFrame): dataframe containing data.

        Returns:
            (pd.DataFrame):predictions and true labels on testing dataset.
        """
        print("HOLDOUT process ...")
        _, test_index = self.stratified_train_test_split(
            data, test_size=self.config.test_size
        )
        train_loader, val_loader, test_loader = self.create_dataloaders(
            data, test_index
        )
        self.run_training_loop(train_loader, val_loader)
        predictions, labels = self.run_evaluation_loop(test_loader)
        evaluation_metrics = self.evaluation(predictions, labels)
        results = pd.DataFrame(
            {
                "predictions": [list(np.concatenate(predictions))],
                "targets": [labels],
                "rmse": evaluation_metrics["rmse"],
                "rmsle": evaluation_metrics["rmsle"],
                "r2": evaluation_metrics["r2"],
            }
        )
        return results

    def run_kfold(self, data: pd.DataFrame):
        """Train and evaluate a model using leave one out cross validation.

        Args:
            data (pd.DataFrame): dataframe containing data.

        Returns:
            (pd.DataFrame): predictions and true labels for each samples on all subjects.
        """
        results = []
        kf = model_selection.KFold(
            n_splits=self.config.num_folds, shuffle=True, random_state=42
        )
        for i, (_, test_index) in enumerate(kf.split(data)):
            print("Fold - {}".format(i))
            train_loader, val_loader, test_loader = self.create_dataloaders(
                data, test_index
            )
            self.run_training_loop(train_loader, val_loader)
            predictions, labels = self.run_evaluation_loop(test_loader)
            evaluation_metrics = self.evaluation(predictions, labels)
            df = pd.DataFrame(
                {
                    "fold": i,
                    "predictions": [list(np.concatenate(predictions))],
                    "targets": [labels],
                    "rmse": evaluation_metrics["rmse"],
                    "rmsle": evaluation_metrics["rmsle"],
                    "r2": evaluation_metrics["r2"],
                }
            )
            results.append(df)
        result_df = pd.concat(results)
        return result_df

    def stratified_train_test_split(
        self, data: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.Series, pd.Series]:
        """Perform a stratified train test split at subject and label level.

        Args:
            data (pd.DataFrame): data to split (sample level).
            test_size (int): ratio of test size.

        Returns:
            (Tuple[pd.Series, pd.Series]): train and test subject ids.
        """
        train, test = model_selection.train_test_split(data, test_size=test_size)
        train_index = train.index.tolist()
        test_index = test.index.tolist()
        return train_index, test_index

    def create_dataloader(self, data: pd.DataFrame) -> DataLoader:
        """Create a pytorch dataloader for provided data.

        Args:
            data (pd.DataFrame): data to convert to dataloader
            training (bool): whether or not you creating training dataloader
            (if training feature augmentation applied).

        Returns:
            (DataLoader): pytorch dataloader.
        """
        return DataLoader(
            PPGDataloader(data),
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def create_dataloaders(
        self, data: pd.DataFrame, test_index: List[str]
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create dataloader for each split (train, test, val).

        Args:
            data (pd.DataFrame): sample level data to split.
            test_index (List[str]): list containing test index.

        Returns:
            (Tuple[DataLoader,DataLoader,DataLoader]): train, val, test dataloaders.
        """

        data_train = data.loc[~data.index.isin(test_index)]
        train_index, val_index = self.stratified_train_test_split(
            data_train, test_size=self.config.test_size_val
        )

        train_loader = self.create_dataloader(
            data_train.loc[data_train.index.isin(train_index)].reset_index()
        )
        val_loader = self.create_dataloader(
            data_train.loc[data_train.index.isin(val_index)].reset_index()
        )
        test_loader = self.create_dataloader(
            data.loc[data.index.isin(test_index)].reset_index()
        )

        return train_loader, val_loader, test_loader

    def collate_fn(
        self, batch: Tuple[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function for pytorch dataloader.

        This function will prepare the batck and stack the features into a tensor.

        Args:
            batch (Tuple[str, torch.Tensor]): batch containing spectral features to load and labels

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): transformed features and labels.
        """
        feature_list = []
        labels_list = []
        for sample in batch:
            feature_list.append(sample[0])
            labels_list.append(sample[1])
        feature = np.stack(feature_list)
        transformed_data = torch.Tensor(feature)
        labels = torch.Tensor(labels_list)
        return transformed_data, labels

    def run_training_loop(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
    ):
        """Run a full training loop.

        Args:
            train_loader (DataLoader): training data
            val_loader (Optional[DataLoader]): optional validation dataset, if passed loop will use early stopping.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )
        loss_fn = torch.nn.MSELoss()
        best_epoch = 0
        min_val_loss = np.inf
        for epoch in range(self.config.n_epochs):
            optimizer.zero_grad()
            self.model.train()
            for batch_idx, batch in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):
                inputs = batch[0].unsqueeze(1)
                if batch_idx == 0:
                    print(inputs.shape)
                outputs = self.model(inputs)
                loss = loss_fn(
                    outputs.to(torch.float64), batch[1].to(torch.float64).view(-1, 1)
                )
                loss /= self.config.gradient_accumulation_steps
                loss.backward()
                if ((batch_idx + 1) % self.config.gradient_accumulation_steps == 0) or (
                    batch_idx + 1 == len(train_loader)
                ):
                    optimizer.step()
                    optimizer.zero_grad()
            if val_loader:
                self.model.eval()
                val_loss = 0
                for val_batch in val_loader:
                    val_inputs = val_batch[0].unsqueeze(1)
                    outputs = self.model(val_inputs)
                    val_loss += loss_fn(
                        outputs.to(torch.float64),
                        val_batch[1].to(torch.float64).view(-1, 1),
                    ).detach()
                epoch_loss = val_loss / len(val_loader)
                if epoch_loss < min_val_loss:
                    best_epoch = epoch
                    min_val_loss = epoch_loss
                print(f"Epoch: {epoch}, Validation loss: {round(float(epoch_loss), 4)}")
                if (epoch - best_epoch) >= self.config.early_stopping_tolerance:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

    def run_evaluation_loop(self, test_loader: DataLoader) -> Tuple[list, list]:
        """Get predictions and labels from test dataset.

        Args:
            test_loader (DataLoader): test dataset

        Returns:
            (Tuple[list, list]): predictions and labels.
        """
        self.model.eval()
        predictions = []
        labels = []
        print(len(test_loader))
        for batch in test_loader:
            predictions += self.model(batch[0].unsqueeze(1)).detach().tolist()
            labels += batch[1].detach().tolist()
        return predictions, labels

    def evaluation(self, y_pred: np.array, y_test: np.array) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            y_pred (np.array): predictions of the model.
            y_test (np.array): true labels.

        Returns:
            Dict[str, float]: dictionary containing evaluation metrics.
        """
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmsle = np.log(rmse)
        r2 = metrics.r2_score(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        return {"rmse": rmse, "mae":mae, "rmsle": rmsle, "r2": r2}
