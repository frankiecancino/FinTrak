from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense
from matplotlib import pyplot, axis
import numpy

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, timesteps):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], timesteps, 1)
    model = Sequential()
    model.add(LSTM(50, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=False, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, len(X), 1)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# run a repeated experiment
def experiment(repeats, series, timesteps):
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, timesteps)
    supervised_values = supervised.values[timesteps:, :]
    # split data into train and test-sets
    n_train = (0.995 * len(supervised_values))
    n_test = (0.005 * len(supervised_values))
    n_train = int(n_train)
    n_test = int(n_test)
    train, test = supervised_values[:n_train, :], supervised_values[-n_test:, :]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # run experiment
    for r in range(repeats):
        # fit the base model
        # lstm_model = fit_lstm(train_scaled, 1, 1, timesteps)
        # load json and create model

        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        lstm_model = model_from_json(loaded_model_json)
        # load weights into new model
        lstm_model.load_weights("model.h5")

        # forecast test dataset
        predictions = list()
        for i in range(len(test_scaled)):

            # predict
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
            if yhat < 0:
                yhat = 0
            # store forecast
            predictions.append(yhat)

    # model_json = lstm_model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # lstm_model.save_weights("model.h5")

    pyplot.plot(predictions, label="Forecast")
    pyplot.ylabel('Amount')
    pyplot.xlabel('Transaction')
    pyplot.legend()
    pyplot.show()

    return predictions


# execute the experiment
def run():
    # load dataset
    series = read_csv('transactions.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    # experiment
    repeats = 1
    df = DataFrame()
    # run experiment
    timesteps = 1
    results = experiment(repeats, series, timesteps)
    # save results
    # results.to_csv('experiment_timesteps_1.csv', index=False)


# entry point
if __name__ == '__main__':
    run()
