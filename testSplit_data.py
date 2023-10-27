
import pandas as pd
import numpy as np


def split_data(data, timesteps):
    Y_units = len(data) // timesteps
    test_units = Y_units * 10 // 100
    eval_units = Y_units * 20 // 100
    train_units = Y_units * timesteps

    test_data = data[-test_units * timesteps:]
    eval_data = data[-(test_units + eval_units) * timesteps:-test_units * timesteps]
    train_data = data[-train_units:-(test_units + eval_units) * timesteps]

    return train_data, eval_data, test_data, train_units

def split_data2(data, timesteps):
    remainder = len(data) % timesteps
    if(remainder >  0):
        data2 = data[remainder:]
    Y_units = len(data2) // timesteps
    test_units = Y_units * 10 // 100
    eval_units = Y_units * 20 // 100
    train_size = (Y_units - (test_units + eval_units)) * timesteps

    test_data = data2[-test_units * timesteps:]
    eval_data = data2[-(test_units + eval_units) * timesteps:-test_units * timesteps]
    train_data = data2[:-(test_units + eval_units) * timesteps]

    return train_data, eval_data, test_data, train_size

# Test Function
def test_function(split_data_function):
    # Sample data now as a numpy array
    close_price_scaled = np.array([i for i in range(2516)]).reshape(-1, 1)

    # Testing
    train, evaluation, test, train_size = split_data_function(close_price_scaled, 60)

    assert len(test) == 240, f"Expected 240 but got {len(test)}"
    assert len(evaluation) == 480, f"Expected 480 but got {len(evaluation)}"
    assert len(train) == 1740, f"Expected 1740 but got {len(train)}"
    assert len(train_size) == 1740, f"Expected 1740 but got {len(train_size)}"


test_function(split_data2)

# Example usage:
# close_price_scaled = pd.Series([i for i in range(2516)])  # Just an example data, replace with your data
# train_data, evaluation_data, test_data = split_data(close_price_scaled, timesteps=60)
#
# print("Train Data Length:", len(train_data))
# print("Evaluation Data Length:", len(evaluation_data))
# print("Test Data Length:", len(test_data))
