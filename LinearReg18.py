import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
import os
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# changes for this version 18 Sept 19,2023
# Purpose: Try to put all the values into a structure so that
#          it can be grouped into a centralized area for easy visual inspection.
# Sept 23,2023. Version LinearReg19.py. Changes are:
#   1.- Use validation_split = 0.2 in model.fit()
#   2.- Rename to evaluate_model_errors() for naming clarity


# Set up logging
logging.basicConfig(level=logging.INFO)

# Global variables
DATA_RATIO = 0.9
INPUT_DATA_FILE = 'AAPL_2013_2023_10_04_v2.csv'
#INPUT_DATA_FILE = 'AAPL_2013_2023_10_04.csv'
#INPUT_DATA_FILE = 'AAPL_2000_2023_10_04.csv'
#INPUT_DATA_FILE = 'AAPL_00_2023_10_12.csv'
INPUT_DATA_PATH = "input/"
TIME_STEPS = 60
N_FUTURE = 30
USED_TRAINED_MODEL = False
MODEL_FILE_BASE = "model_LSTM" + "AAPL-USD_"
EVALUATION_ERROR_FILE = "evaluation_errors.csv"
OUTPUT_RESULTS_PATH = "output/results/"
OUTPUT_MODELS_PATH = "output/model/"

def get_latest_model_filename(base_filename):
    """
    Find the latest model file based on the suffix.
    """
    existing_files = [f for f in os.listdir() if f.startswith(base_filename)]

    if not existing_files:  # No files found
        return None

    # Extract the suffix numbers
    suffixes = [int(f.split('_')[-1].split('.keras')[0]) for f in existing_files]
    latest_suffix = max(suffixes)

    return f"{base_filename}_{latest_suffix}.keras"

def next_available_filename(base_filename):
    """
    Returns the next available filename by appending an incremental number.
    """
    counter = 1
    while True:
        filename = f"{base_filename}_{counter}.keras"
        if not os.path.exists(filename):
            return filename
        counter += 1
def save_trained_model(model, base_filename):
    """
    Save the trained model to the desired location. If file exists, use an alternative name.
    """
    filename = next_available_filename(base_filename)
    model.save(OUTPUT_MODELS_PATH+filename)
    logging.info(f"Model saved to {OUTPUT_MODELS_PATH+filename}")

def load_trained_model(filename):
    """
    Load a trained model from the given filename.
    """
    return tf.keras.models.load_model(filename)

def create_data(data,look_back):
    X, Y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back: i, 0])
        Y.append(data[i, 0])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    Y = np.reshape(Y, (Y.shape[0], 1))
    return X, Y
def generate_forecasts(n_future, X, Y, model, scaler):
    """
    Generates multi-step forecasts using a trained model.

    Parameters:
    - n_future (int): Number of future time steps to predict.
    - X (np.ndarray): Input sequences for the model.
    - Y (np.ndarray): Corresponding target values for the input sequences.
    - model (tf.keras.Model): Trained forecasting model.
    - scaler (scikit-learn scaler): Scaler used to scale the data.

    Returns:
    - y_future_rescaled (np.ndarray): Forecasted values in the original scale.
    """

    y_future = []

    x_pred = X[-1:, :, :]  # last observed input sequence
    y_pred = Y[-1]  # last observed target value

    for _ in range(n_future):
        # feed the last forecast back to the model as an input
        x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

        # generate the next forecast
        y_pred = model.predict(x_pred)

        # save the forecast
        y_future.append(y_pred.flatten()[0])

    # transform the forecasts back to the original scale
    y_future_array = np.array(y_future).reshape(-1, 1)
    y_future_rescaled = scaler.inverse_transform(y_future_array)

    return y_future_rescaled
def load_data(file_name):
    df = pd.read_csv(file_name).dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    logging.info(f"Data loaded successfully-{file_name}")
    return df


def preprocess_data(df):
    close_prices = df[["Date", "Close"]].copy()
    close_prices.columns = ["ds", "y"]
    close_prices.set_index("ds", inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices_scaled = scaler.fit_transform(close_prices)
    logging.info("Data normalized.")

    return close_prices, close_prices_scaled, scaler


def split_data(data, ratio, timesteps):
    remainder = len(data) % timesteps
    if (remainder > 0):
        data = data[remainder:]
    data_units = len(data) / timesteps
    train_units = int(data_units * ratio)
    test_units = data_units - train_units
    train_size = train_units * timesteps
    train_data = data[:train_size]
    test_data = data[train_size:]
    logging.info(f"Training data size: {train_size} / {len(data)}")
    return train_size, train_data, test_data

# Define hyperparameters in a dictionary
LSTM_CONFIG = {
    "first_layer": {
        "units": 94,
        "return_sequences": True
    },
    "second_layer": {
        "units": 128,
        "return_sequences": True
    },
    "third_layer": {
        "units": 64,
        "return_sequences": False
    },
    "dropout_values": [0.2, 0.1, 0.2]
}

def build_model(input_shape):
    model = tf.keras.models.Sequential()

    # First LSTM layer
    model.add(LSTM(units=LSTM_CONFIG["first_layer"]["units"],
                   return_sequences=LSTM_CONFIG["first_layer"]["return_sequences"],
                   input_shape=input_shape))
    model.add(Dropout(LSTM_CONFIG["dropout_values"][0]))

    # Second LSTM layer
    model.add(LSTM(units=LSTM_CONFIG["second_layer"]["units"],
                   return_sequences=LSTM_CONFIG["second_layer"]["return_sequences"]))
    model.add(Dropout(LSTM_CONFIG["dropout_values"][1]))

    # Third LSTM layer
    model.add(LSTM(units=LSTM_CONFIG["third_layer"]["units"],
                   return_sequences=LSTM_CONFIG["third_layer"]["return_sequences"]))
    model.add(Dropout(LSTM_CONFIG["dropout_values"][2]))

    model.add(tf.keras.layers.BatchNormalization(synchronized=True))
    # Dense layer
    model.add(Dense(units=1))

    #model.compile(optimizer="adam", loss="mean_absolute_percentage_error")
    model.compile(optimizer="adam", loss="mean_squared_logarithmic_error")
    #model.compile(optimizer="adam", loss=directional_loss)

    return model

def train_model(model, x_train_data, y_train_label, x_val_data, y_val_label):
    early_stop = EarlyStopping(monitor="val_loss", patience=10)
    # Used 20% of train data for validation
    # history = model.fit(x_train_data, y_train_label, epochs=50, batch_size=32, validation_data=(x_val_data, y_val_label),
    #                     validation_split=0.2, callbacks=[early_stop])
    history = model.fit(x_train_data, y_train_label, epochs=50, batch_size=32,
                        validation_split=0.2, callbacks=[early_stop])
    return history

def evaluate_model_errors(Y_test, y_pred):
    mse = mean_squared_error(Y_test[0], y_pred)
    msle = mean_squared_log_error(Y_test[0], y_pred)
    mae = mean_absolute_error(Y_test[0], y_pred)
    r2 = r2_score(Y_test[0], y_pred)

    print('MSE: ', mse)
    print('MSLE: ', msle)
    print('MAE: ', mae)
    print('R-squared: ', r2)

    # Save the results to CSV
    save_evaluation_to_csv(mse, msle, mae, r2)

def save_evaluation_to_csv(mse, msle, mae, r2):
    # Creating the DataFrame
    evaluation_df = pd.DataFrame({
        'Evaluation errors': ['MSE', 'MSLE', 'MAE', 'R-squared'],
        'Value': [mse, msle, mae, r2]
    })
    # Saving the DataFrame to a CSV file
    evaluation_df.to_csv(OUTPUT_RESULTS_PATH+EVALUATION_ERROR_FILE, index=False)
    logging.info(f"Evaluation errors saved to {OUTPUT_RESULTS_PATH}{EVALUATION_ERROR_FILE}")

def plot_predictions(df, Y_test, y_pred, y_future, n_future):
    # pick up the last date in the "Date" column as the start_day for futures
    last_date = df["Date"][-1:].iloc[-1]
    future_dates = pd.date_range(last_date, periods=n_future + 1)[:-1]

    plt.figure(figsize=(10, 6))
    plt.style.use('fivethirtyeight')
    plot_len = len(Y_test.flatten()) # pick up enough 'Date" cells matched the len of Y_test array only.
    plt.plot(df["Date"][-plot_len:], Y_test.flatten(), label="Actual", linewidth=3, alpha=0.4)
    plt.plot(df["Date"][-plot_len:], y_pred.flatten(), label="Predicted", linewidth=1.5, color='blue')
    plt.plot(future_dates, y_future.flatten(), label="Future", linewidth=1.5, color='orange')
    plt.title("LSTM: Actual vs Predicted vs Future")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Setting the filename as per the given rules
    base_name = INPUT_DATA_FILE.split('.')[0]
    file_name = f"{OUTPUT_RESULTS_PATH}{base_name}_graph_{n_future}.png"
    # Saving the DataFrame to a CSV file
    plt.savefig(file_name)
    plt.show()


def save_forecast_to_csv(close_prices_df, y_future, n_future):
    # Extracting the last date
    last_date = close_prices_df.index[-1]
    next_date = last_date + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=next_date, periods=n_future)
    # Generating future dates
    # future_dates = [last_date + datetime.timedelta(days=i) for i in range(N_FUTURE)]

    # Creating the DataFrame
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Close': y_future.flatten()})

    # Setting the filename as per the given rules
    base_name = INPUT_DATA_FILE.split('.')[0]
    file_name = f"{OUTPUT_RESULTS_PATH}{base_name}_fut_{n_future}.csv"

    # Saving the DataFrame to a CSV file
    forecast_df.to_csv(file_name, index=False)
    logging.info(f"Forecasted data saved to {file_name}")


def main():
    df = load_data(INPUT_DATA_PATH + INPUT_DATA_FILE)
    close_prices, close_prices_scaled, scaler = preprocess_data(df)
    train_size, train_data, test_data = split_data(close_prices_scaled, DATA_RATIO, TIME_STEPS)
    X_train, Y_train = create_data(train_data, TIME_STEPS)
    X_test, Y_test = create_data(test_data, TIME_STEPS)

    # Before training
    if USED_TRAINED_MODEL:
        latest_model_filename = get_latest_model_filename(MODEL_FILE_BASE)
        if latest_model_filename:
            model = load_trained_model(latest_model_filename)
            logging.info(f"Using pre-trained model: {latest_model_filename} for predictions.")
        else:
            logging.error("No pre-trained model found!")
            return
    else:
        model = build_model((X_train.shape[1], 1))
        history = train_model(model, X_train, Y_train, X_test, Y_test)
        print('History of the trained model:', history.history)
        logging.info(f"History of the trained model: {history.history } ")
        save_trained_model(model, MODEL_FILE_BASE)  # Save the model after training

    y_pred = model.predict(X_test)
    y_future = generate_forecasts(N_FUTURE, X_test, y_pred, model, scaler)

    y_pred_rescaled = scaler.inverse_transform(y_pred)

    y_test_rescaled = scaler.inverse_transform([Y_test])

    evaluate_model_errors(y_test_rescaled, y_pred_rescaled)
    save_forecast_to_csv(close_prices, y_future, N_FUTURE)
    plot_predictions(df, y_test_rescaled, y_pred_rescaled, y_future, N_FUTURE)


if __name__ == "__main__":
    main()
