import os 

from ppg_project.training_pipeline import XGBoostTrainingPipeline
from ppg_project.dtypes import XGBoostTrainingPipelineConfig


def run_training_pipeline():
    """Run the training pipeline and apply on the test data for getting the predictions."""
    config = XGBoostTrainingPipelineConfig(
        validation_strategy="prediction_on_test",
        data_path="data",
        features_list = ("features","peak","spectrogram"),
    )
    print(config)
    training_pipeline = XGBoostTrainingPipeline(config)
    df = training_pipeline.run()
    
    if not os.path.exists("output"):
        os.makedirs("output")
    df.to_csv("output/test_prediction.csv", index=False)


if __name__ == "__main__":
    run_training_pipeline()
