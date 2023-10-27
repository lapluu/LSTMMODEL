import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
import os
from keras_tuner import RandomSearch, Hyperband, BayesianOptimization
from tensorflow import keras
import keras_tuner as kt
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime

from tensorflow.python.estimator import keras
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dropout, Dense

#Global variables area
INPUT_DATA_FILE = 'input/AAPL_2013_2023_10_04.csv'
SELECTED_TUNER = "RandomSearch"  # Options: "RandomSearch", "Hyperband", "BayesianOptimization"
LOOK_BACK = 60
N_FUTURE = 10
SPLIT_RATIO = 0.8
MODEL_FILE_BASE = "model_SRCH_LSTM"

logging.basicConfig(level=logging.INFO)


def save_hyperparameters_to_file(hps, filename):
    with open(filename, 'w') as f:
        json.dump(hps.get_config(), f)


def load_and_preprocess_data(filename):
    df = pd.read_csv(filename).dropna()
    logging.info("Data loaded successfully.")
    df["Date"] = pd.to_datetime(df["Date"])
    close_prices = df[["Date", "Close"]].copy()
    close_prices.columns = ["ds", "y"]
    close_prices.set_index("ds", inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices_scaled = scaler.fit_transform(close_prices)
    logging.info("Data normalized.")
    return df, close_prices, close_prices_scaled, scaler


def split_data(data, split_ratio):
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    logging.info(f"Training data size: {train_size} / {len(data)} ")
    return train_data, test_data, train_size


def create_dataset(data, look_back=60):
    X, Y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back: i, 0])
        Y.append(data[i, 0])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, Y


def build_model(hp, input_shape):
    model = tf.keras.models.Sequential()
    model.add(LSTM(units=hp.Int('units_input', min_value=32, max_value=128, step=32),
                   return_sequences=True, input_shape=input_shape))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_hidden', min_value=32, max_value=128, step=32), return_sequences=True))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_output', min_value=32, max_value=128, step=32)))
    model.add(Dropout(rate=hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int("units", min_value=32, max_value=128, step=32),
                    activation="relu"))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.mean_squared_logarithmic_error)
    return model


def get_tuner(name, input_shape):
    if name == "RandomSearch":
        return RandomSearch(lambda hp: build_model(hp, input_shape), objective='val_loss', max_trials=5, executions_per_trial=3, directory='project/RandomSearch/', overwrite=False, project_name='StockPriceLSTM')
    elif name == "Hyperband":
        return Hyperband(lambda hp: build_model(hp, input_shape), objective='val_loss', max_epochs=50, directory='project/Hyperband/', overwrite=False,project_name='StockPriceLSTM 2')
    elif name == "BayesianOptimization":
        return BayesianOptimization(lambda hp: build_model(hp, input_shape), objective='val_loss', max_trials=10, directory='project/BayesianOptimization/', overwrite=False, project_name='StockPriceLSTM')
    else:
        raise ValueError(f"Unsupported tuner: {name}")

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

def create_future_dates(yyyymmdd, n_future):
    date = pd.to_datetime(yyyymmdd)
    next_date = date + pd.Timedelta(days=1)
    future_days =  pd.DataFrame(pd.date_range(start=next_date, periods=n_future ),
                                columns=['Date'])
    return future_days

def plot_predictions(df, train_size,n_future, look_back, Y_test, y_pred,y_future):
    last_date = df["Date"][train_size + look_back:].iloc[-1]
    future_dates = create_future_dates(last_date, n_future)
    plt.figure(figsize=(10, 6))
    plt.style.use('fivethirtyeight')
    plt.plot(df["Date"][train_size + look_back:], Y_test.flatten(), label="Actual", linewidth=3, alpha=0.4)
    plt.plot(df["Date"][train_size + look_back:], y_pred.flatten(), label="Predicted", linewidth=1.5, color='blue')
    plt.plot(future_dates, y_future.flatten(), label="Future", linewidth=1.5, color='orange')
  #plt.plot(df["Date"][train_size + look_back:], y_future.flatten(), label="Future", linewidth=1.5, color='green')

    plt.title("LSTM: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_forecast_to_csv(close_prices_df, y_future, n_future):
    # Extracting the last date
    last_date = close_prices_df.index[-1]
    next_date = last_date + pd.Timedelta(days=1)

    # Generating future dates
    future_dates = pd.date_range(start=next_date, periods=n_future)
    # Creating the DataFrame
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Close': y_future.flatten()})

    # Setting the filename as per the given rules
    base_name = INPUT_DATA_FILE.split('.')[0]
    file_name = f"{base_name}_fut_{N_FUTURE}.csv"

    # Saving the DataFrame to a CSV file
    forecast_df.to_csv(file_name, index=False)
    logging.info(f"Forecasted data saved to {file_name}")


def evaluate_model(Y_test, y_pred):
    mse = mean_squared_error(Y_test[0], y_pred)
    msle = mean_squared_log_error(Y_test[0], y_pred)
    mae = mean_absolute_error(Y_test[0], y_pred)
    r2 = r2_score(Y_test[0], y_pred)

    print('MSE: ', mse)
    print('MSLE: ', msle)
    print('MAE: ', mae)
    print('R-squared: ', r2)

def train_model(model, x_train_data, y_train_label, x_val_data, y_val_label):
    early_stop = EarlyStopping(monitor="val_loss", patience=10)
    history = model.fit(x_train_data, y_train_label, epochs=50, batch_size=64, validation_data=(x_val_data, y_val_label),
                        callbacks=[early_stop])
    return history

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
    try:
        model.save(filename)
    except Exception as e:
        print("Error saving the model:", str(e))
        logging.info(f"Error saving the model: {e}")

    logging.info(f"Model saved to {filename}")


def main():

    df, close_prices, close_prices_scaled, scaler = load_and_preprocess_data(INPUT_DATA_FILE)
    train_data, test_data, train_size = split_data(close_prices_scaled,SPLIT_RATIO)

    logging.info(f"Look back: {LOOK_BACK}")
    # created the dataframe into data and label portion.
    X_train, Y_train = create_dataset(train_data, LOOK_BACK)
    X_test, Y_test = create_dataset(test_data, LOOK_BACK)

    tuner = get_tuner(SELECTED_TUNER, (X_train.shape[1], 1))
    tuner.search_space_summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=10)
    tuner.search(X_train, Y_train, epochs=20, batch_size=32,
                 validation_data=(X_test, Y_test), callbacks=[early_stop])
    tuner.search_space_summary(extended=False)
    # Retrieve and print the best two models.
    tuner_best_model = tuner.get_best_models(num_models=1)[0]
    print("Best model class:", tuner_best_model.__class__.__name__)
    #tuner_best_model.build((X_train.shape[1], 1))
    tuner_best_model.summary()
    y_pred = tuner_best_model.predict(X_test)
    y_future_rescaled = generate_forecasts(N_FUTURE, X_test, y_pred, tuner_best_model, scaler)

    y_pred_rescaled = scaler.inverse_transform(y_pred)
    Y_test_rescaled = scaler.inverse_transform([Y_test])

    evaluate_model(Y_test_rescaled, y_pred_rescaled)
    plot_predictions(df, train_size, N_FUTURE, LOOK_BACK, Y_test, y_pred, y_future_rescaled)


    # Retrieve and print the best hyperparameters.
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:", best_hps.get_config())

    # saved the hyperparameters to file.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hyperparameters_{timestamp}.json"
    save_hyperparameters_to_file(best_hps, filename)

    # Now, build the model from scratch with best_hps; rather than retrieved from the tuner
    # train it again on the data for 50 epochs
    built_model = build_model(best_hps, (X_train.shape[1], 1))
    history = train_model(built_model, X_train, Y_train, X_test, Y_test)
    logging.info(f"History of the trained model: {history.history} ")
    built_model.summary()
    # predict with the built model; not the tuner's model
    y_pred2 = built_model.predict(X_test)
    y_future2_rescaled = generate_forecasts(N_FUTURE, X_test, y_pred2, built_model, scaler)
    y_pred2_rescaled = scaler.inverse_transform(y_pred2)
    # evaluation of the built_model
    evaluate_model(Y_test_rescaled, y_pred2_rescaled)
    plot_predictions(df, train_size, N_FUTURE, LOOK_BACK, Y_test, y_pred, y_future2_rescaled)


if __name__ == "__main__":
    main()
