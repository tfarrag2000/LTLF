from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, GRU
from keras.callbacks import EarlyStopping
from pandas import ExcelWriter
import uuid
import numpy as np
import os

os.environ["PATH"] += os.pathsep + r'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.utils import plot_model

# experiment parameters
parameters = dict()


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# def removing_seasonal_data(data, columnname, dropnan=True):
#     from stldecompose import decompose
#     # https://stackoverflow.com/questions/20672236/time-series-decomposition-function-in-python
#     stl = decompose(data[columnname])
#     stlplot = stl.plot()
#     # save to file
#     stlplot.savefig("stl_seasonal_1.png")
#     data[columnname] = stl.resid
#     data.dropna(inplace=dropnan)
#     return data


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def load_prepare_data():
    # load dataset
    dataset = read_csv('load.csv', header=0, index_col=0, parse_dates=True)
    # dataset = removing_seasonal_data(dataset, 'MaxLoad')
    values = dataset.values
    print(dataset)
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, parameters["n_days"], 1, dropnan=True)

    return reframed, scaler


def plotting_save_experiment_data(model, history, y_actual, y_predicted):
    print(model.summary())
    # save data
    writer = ExcelWriter('experimentOutput\\' + parameters["ID"] + 'results.xlsx')
    df = DataFrame.from_dict(parameters, orient='index')
    df.to_excel(writer, 'parameters')
    df = DataFrame(list(zip(y_actual, y_predicted)), columns=['y_actual', 'y_predicted'])
    df.to_excel(writer, 'predicted')
    df = DataFrame(list(zip(history.history['loss'], history.history['val_loss'])), columns=['loss', 'val_loss'])
    df.to_excel(writer, 'loss_history')
    writer.save()
    writer.close()

    # plot history
    plot_model(model, to_file='experimentOutput\\' + parameters["ID"] + 'model_fig.png', show_shapes=True,
               show_layer_names=True)

    # plot history loss
    pyplot.close()
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.savefig('experimentOutput\\' + parameters["ID"] + "loss_fig.png")
    pyplot.close()

    # calculate RMSE
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    print('Test RMSE: %.3f' % rmse)
    # calculate MAPE
    MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
    print('Test MAPE: %.3f' % MAPE)

    # plot actual vs predicted
    pyplot.plot(y_actual, label='actual')
    pyplot.plot(y_predicted, label='predicted')
    pyplot.legend()
    pyplot.title('RMSE: %.3f' % rmse + " , " + 'MAPE: %.3f' % MAPE)
    pyplot.savefig('experimentOutput\\' + parameters["ID"] + "forcast_fig.png")
    pyplot.close()


def create_fit_model(data, scaler):
    # split into train and test sets
    values = data.values
    print(values.shape)
    train = values[:parameters["n_traindays"], :]
    test = values[parameters["n_traindays"]:, :]
    # split into input and outputs
    n_obs = parameters["n_days"] * parameters["n_features"]
    train_X, train_y = train[:, :n_obs], train[:, -parameters["n_features"]]
    test_X, test_y = test[:, :n_obs], test[:, -parameters["n_features"]]
    # print(train_X.shape, len(train_X), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], parameters["n_days"], parameters["n_features"]))
    test_X = test_X.reshape((test_X.shape[0], parameters["n_days"], parameters["n_features"]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(parameters["n_neurons"], input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(parameters["n_neurons"], return_sequences=True))
    model.add(LSTM(parameters["n_neurons"], return_sequences=True))
    model.add(LSTM(parameters["n_neurons"]))

    # model.add(Dense(60))
    # #model.add(Dropout(0.2))
    # model.add(Dense(20))
    # # model.add(Dropout(0.2))
    # # model.add(Dense(10))
    # # model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    clbs = None
    print(model.summary())

    if parameters["earlystop"]:
        earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')
        clbs = [earlyStopping]

    history = model.fit(train_X, train_y, epochs=parameters["n_epochs"], batch_size=parameters["n_batch"],
                        validation_data=(test_X, test_y),
                        verbose=parameters["model_train_verbose"],
                        shuffle=False, callbacks=clbs)

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], parameters["n_days"] * parameters["n_features"]))

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -parameters["n_features"] + 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -parameters["n_features"] + 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    plotting_save_experiment_data(model, history, inv_y, inv_yhat)


def run_experiment():
    data, scaler = load_prepare_data()
    create_fit_model(data, scaler)


def main():
    parameters["ID"] = uuid.uuid4().hex
    parameters["n_days"] = 7
    parameters["n_features"] = 4
    parameters["n_traindays"] = 365 * 11
    parameters["n_epochs"] = 1
    parameters["n_batch"] = 128
    parameters["n_neurons"] = 10
    parameters["model_train_verbose"] = 2
    parameters["earlystop"] = True

    run_experiment()


main()
