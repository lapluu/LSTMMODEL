Aug 26,2023.
The LinearReg9.py is the best fit for the stock prediction so far.
Haven't try to use the tuning program such as LinearRegFinal5.py as an example.

- Will create a LinearReg10.py which adding new forcast function
to predict the future stock price.
**************Here is the summary of the model:
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=64))
model.add(Dropout(0.2))

model.add(Dense(units=1))
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), callbacks=[early_stop])
***********************