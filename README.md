# TimeSeriesPrediction
Hybrid CNN–GP Attention Model for Electrical Load Forecasting
1. Overview

This repository contains an implementation of a deep learning framework for electrical load forecasting, integrating Convolutional Neural Networks (CNNs) with a Gaussian Process (GP)-enhanced attention mechanism. The code is designed to handle time-series data preprocessing, feature extraction (in time, frequency, and wavelet domains), and model training for predicting short- to medium-term electricity demand.
2. Key Features

    Data Preprocessing
        KNN-based imputation for missing data.
        Isolation Forest for outlier detection.
        First-order differencing for trend removal and stationarity.
    Feature Extraction
        Time Domain: Statistical measures (mean, variance, skewness), zero-crossing rate, mean absolute deviation, Hjorth parameters.
        Frequency Domain: Spectral analysis, roll-off frequency, bandwidth, centroid calculations.
        Wavelet Domain: Multi-level wavelet decomposition and coefficient statistics.
    Model Architecture
        CNN for hierarchical feature extraction from time-series data.
        Attention mechanism augmented with Gaussian Process kernels (GP-Kernel Attention) for improved uncertainty quantification and temporal dependency modeling.
        Positional encoding to preserve the temporal order of data sequences.
    Training Strategy
        Uses the Adam optimizer with early stopping.
        Reduce LR on Plateau to dynamically adjust the learning rate.

3. Installation

    Clone this repository.

git clone https://github.com/your-username/Hybrid-CNN-GP-Attention-LoadForecast.git
cd Hybrid-CNN-GP-Attention-LoadForecast

Create and activate a virtual environment (optional but recommended).

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate.bat  # Windows

Install the required Python packages using requirements.txt.

    pip install -r requirements.txt

4. Usage

    Data Preparation
        Place your time-series load data in the data/ folder (or update file paths in the code).
        Ensure each file has timestamps and load values.

    Preprocessing
        Run the provided preprocessing script (e.g., python preprocess.py) to impute missing values and detect outliers.
        The script will also perform first-order differencing and save the processed features in a preprocessed/ directory.

    Model Training
        Execute python train.py (or the corresponding training script) to start model training.
        Adjust hyperparameters (learning rate, batch size, etc.) in the config.yaml or in a dedicated parameters section.

    Prediction and Evaluation
        After training, use python predict.py to generate future load forecasts.
        Evaluation metrics (MAPE, MSE, R-squared) will be automatically computed if you provide a test dataset.

5. Results

    Expected performance: The paper reports improved accuracy over baseline methods (e.g., LSTM, simple attention, or plain CNN).
    The integrated GP-Kernel Attention mechanism helps capture time-series uncertainty and correlations effectively.
