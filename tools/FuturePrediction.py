import logging
import matplotlib.pyplot as plt
from tensorflow.python.ops.ragged.ragged_conversion_ops import from_tensor
from tqdm import tqdm
from tools.PreProcessing import (extract_features, de_difference,compute_difference,load_csv_data,impute_missing_values,
                                 remove_outliers,normalize_data)
import numpy as np
import os
import pickle
import tensorflow as tf



def future_prediction(model, data, original_data, params, scaler=None, results_dir=None):
    """Perform future prediction using the model."""
    logger = logging.getLogger('TimeSeriesModel')
    logger.info("Performing future prediction")
    predictions = []
    window_size = params['window_size']
    num_features = data.shape[1]
    input_seq = data[-window_size:, :]
    if params.get('use_difference', False):
        last_actual_value = original_data[-1, :]

    # Prediction loop
    for _ in tqdm(range(params['horizon']), desc="Predicting future"):
        features_list = []
        for feature_idx in range(num_features):
            window_feature = input_seq[:, feature_idx].reshape(-1, 1)
            features = extract_features(window_feature, params)
            features_list.append(features)
        combined_features = np.concatenate(features_list)
        input_seq_reshaped = combined_features.reshape((1, -1))
        pred = model.predict(input_seq_reshaped)
        if scaler is not None:
            pred_denorm = scaler.inverse_transform(pred)
            predictions.append(pred_denorm[0])
            new_input = scaler.transform(pred_denorm)
        else:
            predictions.append(pred[0])
            new_input = pred
        input_seq = np.vstack((input_seq[1:], new_input))

    predictions = np.array(predictions)
    if params.get('use_difference', False):
        predictions = de_difference(predictions, last_actual_value)

    # Enhanced plotting section for academic paper
    if params['is_plot']:
        plt.style.use('grayscale') # Use academic-style theme
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)  # Higher DPI for publication quality

        # Plot historical data
        time_historical = np.arange(len(original_data))
        ax.plot(time_historical, original_data,
                label='Historical Data',
                color='#2C3E50',  # Darker blue for better contrast
                linewidth=1.5)

        # Plot predictions
        time_future = np.arange(len(original_data), len(original_data) + len(predictions))
        ax.plot(time_future, predictions,
                label='Future Predictions',
                color='#E74C3C',  # Professional red
                linewidth=1.5,
                linestyle='--')  # Dashed line for predictions

        # Customize the plot
        ax.set_xlabel('Time Steps', fontsize=10, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10, fontweight='bold')
        ax.set_title('Time Series Forecasting Results',
                     fontsize=12,
                     fontweight='bold',
                     pad=15)

        # Enhance legend
        ax.legend(loc='upper left',
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  fontsize=10)

        # Customize grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Tight layout
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(results_dir, "future_prediction.png")
        plt.savefig(save_path,
                    format='png',
                    bbox_inches='tight',
                    pad_inches=0.1)

        if params['is_plot']:
            plt.show()
        plt.close()

    return predictions


def future_prediction_with_new_data(model_path=None, params_path=None, data_path=None):
    """Perform future prediction using new data."""
    root_address = os.path.dirname(os.path.abspath(__file__))
    logger = logging.getLogger('TimeSeriesModel')
    logger.info("Performing future prediction with new data")
    if model_path is None or params_path is None or data_path is None:
        assert "there are conflict in the providing data"
    model = tf.keras.models.load_model(model_path)(model_path)
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    params['results_save_dir'] = os.path.join(root_address, 'results')
    new_data = load_csv_data(data_path, logger)
    if params['imputation_method']:
        new_data = impute_missing_values(
            new_data,
            method=params['imputation_method'],
            n_neighbors=params['imputation_neighbors'],
            logger=logger
        )
    if params.get('use_difference', False):
        new_data = compute_difference(new_data, params=params)
    if params['outlier_removal']:
        new_data = remove_outliers(new_data, method=params['outlier_method'], params=params)
    if params['normalization_method']:
        scaler = params['scaler']
        if scaler:
            new_data = scaler.transform(new_data)
        else:
            new_data, _ = normalize_data(new_data, method=params['normalization_method'], logger=logger)
    else:
        scaler = None
    if len(new_data.shape) == 1 or new_data.shape[1] == 1:
        new_data = new_data.reshape(-1, 1)
    predictions = future_prediction(
        model,
        new_data,
        new_data,
        params,
        scaler,
        results_dir=params['results_save_dir']
    )
    logger.info(f"Future Predictions (new data): {predictions}")
    return predictions

