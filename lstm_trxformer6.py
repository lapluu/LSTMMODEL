import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import layers
import time
import csv
import timeit
import pandas as pd
import datetime
import logging

# version lstm_trxformer4.py
# just added  a plot for the last X days only.
# version lstm_trxformer5.py ==> Change timestep=60

# Global variables
TICKER = 'AAPL'
logging.basicConfig(level=logging.INFO)
TIMESTEPS = 60
# Class and function declaration in here
class ETL:
    """
    ticker: str
    period: string
    test_size: float betwee 0 and 1
    n_input: int
    timestep: int
    Extracts data for stock with ticker `ticker` from yf api,
    splits the data into train and test sets by date,
    reshapes the data into np.array of shape [#weeks, 5, 1],
    converts our problem into supervised learning problem.
    """
    def __init__(self, ticker, test_size=0.2, period='max', n_input=5, timestep=TIMESTEPS) -> None:
        self.ticker = ticker
        self.period = period
        self.test_size = test_size
        self.n_input = n_input
        self.df = self.extract_historic_data()
        self.timestep = timestep
        self.train, self.test = self.etl()
        self.X_train, self.y_train = self.to_supervised(self.train)
        self.X_test, self.y_test = self.to_supervised(self.test)

    def extract_historic_data(self) -> pd.Series:
        """
        gets historical data from yf api.
        """
        t = yf.Ticker(self.ticker)
        history = t.history(period=self.period)
        return history.Close


    def split_data(self) -> tuple:
        """
        Splits our pd.Series into train and test series with
        test series representing test_size * 100 % of data.
        """
        data = self.extract_historic_data()
        if len(data) != 0:
            # Adjust the length of data so that its total length is a multiple of self.timestep
            remainder = len(data) % self.timestep
            if remainder != 0:
                data = data[remainder:]
            # Split the adjusted data into train and test
            train_idx = round(len(data) * (1-self.test_size))
            train = data[:train_idx]
            test = data[train_idx:]
            train = np.array(train)
            test = np.array(test)
            return train[:, np.newaxis], test[:, np.newaxis]
        else:
            raise Exception('Data set is empty, cannot split.')

    def window_and_reshape(self, data) -> np.array:
        """
        Reformats data into shape our model needs,
        namely, [# samples, timestep, # feautures]
        samples
        """
        NUM_FEATURES = 1
        samples = int(data.shape[0] / self.timestep)
        result = np.array(np.array_split(data, samples))
        return result.reshape((samples, self.timestep, NUM_FEATURES))

    def transform(self, train, test) -> np.array:
        train_remainder = train.shape[0] % self.timestep
        test_remainder = test.shape[0] % self.timestep
        if train_remainder != 0 and test_remainder != 0:
            train = train[train_remainder:]
            test = test[test_remainder:]
        elif train_remainder != 0:
            train = train[train_remainder:]
        elif test_remainder != 0:
            test = test[test_remainder:]
        return self.window_and_reshape(train), self.window_and_reshape(test)

    def etl(self) -> tuple[np.array, np.array]:
        """
        Runs complete ETL
        """
        train, test = self.split_data()
        return self.transform(train, test)

    def to_supervised(self, train, n_out=5) -> tuple:
        """
        Converts our time series prediction problem to a
        supervised learning problem.
        """
        # flatted the data
        data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
        X, y = [], []
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + self.n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end <= len(data):
                x_input = data[in_start:in_end, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y.append(data[in_end:out_end, 0])
                # move along one time step
                in_start += 1
        return np.array(X), np.array(y)

class PredictAndForecast:
    """
    model: tf.keras.Model
    train: np.array
    test: np.array
    Takes a trained model, train, and test datasets and returns predictions
    of len(test) with same shape.
    """
    def __init__(self, model, train, test, n_input=5) -> None:
        self.model = model
        self.train = train
        self.test = test
        self.n_input = n_input
        self.predictions = self.get_predictions()

    def forecast(self, history) -> np.array:
        """
        Given last weeks actual data, forecasts next weeks prices.
        """
        # flatten data
        data = np.array(history)
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        # retrieve last observations for input data
        input_x = data[-self.n_input:, :]
        # reshape into [1, n_input, 1]
        input_x = input_x.reshape((1, len(input_x), input_x.shape[1]))
        # forecast the next week
        yhat = self.model.predict(input_x, verbose=0)
        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    def get_predictions(self) -> np.array:
        """
        compiles models predictions week by week over entire
        test set.
        """
        # history is a list of weekly data
        history = [x for x in self.train]
        # walk-forward validation over each week
        predictions = []
        for i in range(len(self.test)):
            yhat_sequence = self.forecast(history)
            # store the predictions
            predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
            history.append(self.test[i, :])
        return np.array(predictions)

class Evaluate:

  def __init__(self, actual, predictions) -> None:
    self.actual = actual
    self.predictions = predictions
    self.var_ratio = self.compare_var()
    self.mape = self.evaluate_model_with_mape()

  def compare_var(self):
    return abs( 1 - (np.var(self.predictions) / np.var(self.actual)))

  def evaluate_model_with_mape(self):
    return mean_absolute_percentage_error(self.actual.flatten(), self.predictions.flatten())

def build_lstm(etl: ETL, epochs=25, batch_size=32) -> tf.keras.Model:
  """
  Builds, compiles, and fits our LSTM baseline model.
  """
  n_timesteps, n_features, n_outputs = TIMESTEPS, 1, 5
  callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
  model = Sequential()
  model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(n_outputs))
  print('compiling baseline model...')
  logging.info('compiling baseline model...')

  model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
  print('fitting model...')
  logging.info('fitting model...')
  start = time.time()
  history = model.fit(etl.X_train, etl.y_train, batch_size=batch_size, epochs=epochs, validation_data=(etl.X_test, etl.y_test), verbose=1, callbacks=callbacks)
  print(time.time() - start)
  return model, history



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0, epsilon=1e-6, attention_axes=None, kernel_size=1):
  """
  Creates a single transformer block.
  """
  x = layers.LayerNormalization(epsilon=epsilon)(inputs)
  x = layers.MultiHeadAttention(
      key_dim=head_size, num_heads=num_heads, dropout=dropout,
      attention_axes=attention_axes
      )(x, x)
  x = layers.Dropout(dropout)(x)
  res = x + inputs

    # Feed Forward Part
  x = layers.LayerNormalization(epsilon=epsilon)(res)
  x = layers.Conv1D(filters=ff_dim, kernel_size=kernel_size, activation="relu")(x)
  x = layers.Dropout(dropout)(x)
  x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=kernel_size)(x)
  return x + res



def build_transfromer(head_size, num_heads, ff_dim, num_trans_blocks, mlp_units, dropout=0, mlp_dropout=0, attention_axes=None, epsilon=1e-6, kernel_size=1):
  """
  Creates final model by building many transformer blocks.
  """
  n_timesteps, n_features, n_outputs = 60, 1, 5
  inputs = tf.keras.Input(shape=(n_timesteps, n_features))
  x = inputs
  for _ in range(num_trans_blocks):
    x = transformer_encoder(x, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout, attention_axes=attention_axes, kernel_size=kernel_size, epsilon=epsilon)

  x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
  for dim in mlp_units:
    x = layers.Dense(dim, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)

  outputs = layers.Dense(n_outputs)(x)
  return tf.keras.Model(inputs, outputs)

def fit_transformer(transformer: tf.keras.Model):
  """
  Compiles and fits our transformer.
  """
  transformer.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["mae", 'mape'])

  callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
  start = time.time()
  hist = transformer.fit(data.X_train, data.y_train, batch_size=32, epochs=25, verbose=1, callbacks=callbacks)
  print(time.time() - start)
  return hist

def get_last_days(test, preds, lastDays):
    last_XDays_test = test[1:]
    last_XDays_preds = preds[1:]


def get_last_Days(test, preds, days):
    # Check if the array length is greater than or equal to xdays
    if len(test) < days or len(preds) < days:
        raise ValueError("The length of the input array is less than the requested number of days.")

    # Return the last days from the array
    return test[-days:], preds[-days:]

def plot_results(test, preds, df, image_path=None, title_suffix=None,
                 xlabel='AAPL stock Price', xdays=0):
  """
  Plots training data in blue, actual values in red, and predictions in green,
  over time.
  """
  fig, ax = plt.subplots(figsize=(20,6))
  # x = df.Close[-498:].index
  if(xdays > 0):
      rows = xdays // test.shape[1] # the whole number portion.
      plot_test2 = test[-rows:]   # only the last "rows" of the array
      plot_preds2 = preds[-rows:]
      x = df[-(rows * test.shape[1]):].index # test.shape[1] contains the number of columns.
  else:
      plot_test2 = test[1:]
      plot_preds2 = preds[1:]
      x = df[-(plot_test2.shape[0]*plot_test2.shape[1]):].index
  plot_test = plot_test2.reshape((plot_test2.shape[0]*plot_test2.shape[1], 1))
  plot_preds = plot_preds2.reshape((plot_preds2.shape[0]*plot_preds2.shape[1], 1))
  ax.plot(x, plot_test, label='actual')
  ax.plot(x, plot_preds, label='preds')
  if title_suffix==None:
    ax.set_title('Predictions vs. Actual')
  else:
    ax.set_title(f'Predictions vs. Actual, {title_suffix}')
  ax.set_xlabel('Date')
  ax.set_ylabel(xlabel)
  ax.legend()
  if image_path != None:
    imagedir = '/content/drive/MyDrive/Colab Notebooks/images'
    plt.savefig(f'{imagedir}/{image_path}.png')
  plt.show()


def output_2_CSV_file(test, preds, df, csv_path):
    """
    Outputs the results to a CSV file with the format:
    Date, Actual Price, Predicted Price
    """
    # Extract dates
    x = df[-(test.shape[0] * test.shape[1]):].index

    # Print shape of test for debugging
    print("Shape of test:", test.shape)

    # Flatten the test and predictions for ease of use
    flattened_size = (test.shape[0] - 1) * test.shape[1]  # adjusted to take slicing into account
    plot_test = test[1:].reshape((flattened_size, 1))
    plot_preds = preds[1:].reshape((flattened_size, 1))

    # Create a new list to store rows
    rows = [["Date", "Actual Price", "Predicted Price"]]

    # Zip together dates, actual prices, and predicted prices
    for date, actual, prediction in zip(x, plot_test, plot_preds):
        rows.append([date.strftime('%d-%m-%Y'), actual[0], prediction[0]])

    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Results saved to {csv_path}")

def save_eval_2_file(mape, var_ratio, csv_path):
    """
    Saves the evaluation results (MAPE & Variance Ratio) to a specified CSV file.

    Parameters:
    - mape (float): Mean Absolute Percentage Error.
    - var_ratio (float): Variance Ratio.
    - csv_path (str): Path to save the CSV file.

    Returns:
    None
    """
    # Create a list to store rows
    rows = [["Metric", "Value"],
            ["MAPE", mape],
            ["Variance Ratio", var_ratio]]

    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Evaluation results saved to {csv_path}")


# Execute code in here
# extract data & prepare historical data by stock ticker
data = ETL(TICKER)
baseline = build_lstm(data)

baseline_model = baseline[0]
history = baseline[1]
# display the model summary
baseline_model.summary()
# build the transformer
transformer = build_transfromer(head_size=128, num_heads=4, ff_dim=2, num_trans_blocks=4, mlp_units=[256], mlp_dropout=0.10, dropout=0.10, attention_axes=1)
# display the summary
transformer.summary()
# fit the transformer
hist = fit_transformer(transformer)
# make predictions using the baseline model first
start = time.time()
baseline_preds = PredictAndForecast(baseline_model, data.train, data.test)
print(time.time() - start)
# make predictions using the transformer
start = time.time()
transformer_preds = PredictAndForecast(transformer, data.train, data.test)
print(time.time() - start)
# evaluate the baseline models
baseline_evals = Evaluate(data.test, baseline_preds.predictions)
# evaluate the transformer models
transformer_evals = Evaluate(data.test, transformer_preds.predictions)
# display the MAPE values of the baseline and transformer models
baseline_evals.mape, transformer_evals.mape
baseline_evals.var_ratio, transformer_evals.var_ratio

save_eval_2_file(baseline_evals.mape, baseline_evals.var_ratio, 'lstm_eval_errors.csv')

save_eval_2_file(transformer_evals.mape, transformer_evals.var_ratio, 'trxformer_eval_errors.csv')

LAST_DAYS = 90 # The last 3 months

plot_results(data.test, baseline_preds.predictions, data.df, title_suffix='LSTM',
             xlabel=TICKER +'Stock Price', xdays = 0 )
test2, preds2 = get_last_Days(data.test, baseline_preds.predictions, LAST_DAYS)
plot_results(test2, preds2, data.df, title_suffix='LSTM2', xlabel=TICKER +'Stock Price', xdays = LAST_DAYS )

plot_results(data.test, transformer_preds.predictions, data.df, title_suffix='Transformer', xlabel=TICKER +'Stock Price')

# ...

# output to csv file for LSTM model
logging.info("output result to lstm_results.csv file for LSTM model.")
output_2_CSV_file(data.test, baseline_preds.predictions, data.df, TICKER + '_lstm_results.csv')

# output to csv file for transformer model
logging.info("output result to trxformer_results.csv for Transformer model.")
output_2_CSV_file(data.test, transformer_preds.predictions, data.df, TICKER +'_trxformer_results.csv')







