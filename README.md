
<h1>PPG Project</h1>

This repository was create for the PPG challenge. To use this code make sure you create a data folder and put inside:train.csv, train_labels.csv and test.csv.
 
<h1>Pipeline</h1>

To analyse the data, I worked on two type of pipeline: 

- training_pipeline.py: a standard pipeline involving different transformation of features which includes spectrogram, peak detection analysis and wavelets. 
- dl_training_pipeline.py: a deep learning pipeline aiming to build a Convolution Neural Network on spectral features (spectorgram, wavelet) using a dataloaders (ppg_dataloaders.py) and deep learning model (pp_models.py).

<h1> Parameters </h1>
All the parameters used are located in dtypes under the different dataclasses:

- XGBoostTrainingPipelineConfig: config for XGB pipeline
- DLTrainingPipelineConfig: config for DL pipeline

<h1> Notebooks </h1>
You will find 5 notebooks in the folder notebooks, you should be able to run all of them:

- test_preprocessing.ipynb: shows some exploratory data analysis I have done.
- test_dl_pipeline.ipynb: goes through the different steps to run the training of the CNN.
- test_xgb_pipeline_hypertuning.ipynb: shows the training of a XGBRegressor pipeline with hypertuning of parameter and retraining with the best parameter. Results are computed over cross_validation.
- test_xgb_pipeline.ipynb: shows the standard pipeline with no hyperparameter tuning. Results are computed over cross_validation.


<h1> Features </h1>
Here is the list of category of features I investigated:
- features: use the crafted features as it is.
- ppg: compute mean and std over the full signal.
- peak: use peak detection and compute multiple features (number of peaks in signal, distance peak to peak (mean,std), peak prominence (mean,std) and peak width (mean,std)).
- spectrogram: mean and std over time for frequencies between [0-5Hz].
- wavelet: mean and std over time for  frequencies between [0-5Hz]. Here I had an issue,it was too computationally expensive and I had to downsample the signal. 

<h1> Scripts </h1>
If you wish to reproduce the results I send, you can simply run run_script.py to get the predictions. Due to lack of time, the deep learning approach didnt bring correct results. The preferred features for the DL approach are wavelets which were two computationaly expensive.  



