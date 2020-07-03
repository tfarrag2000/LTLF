import tensorflow as tf
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Bidirectional
from tensorflow.keras.layers import LSTM, Input, Flatten, Reshape, TimeDistributed, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from pandas import ExcelWriter
import numpy as np
import mysql.connector
import os
from psokeras import Optimizer

from tensorflow.python.keras.layers import Conv1D, MaxPooling1D

os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    dataset = read_csv('Load.csv', header=0, index_col=0, parse_dates=True)

    values = dataset.values
    # print(dataset)
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, parameters["n_days"], 1, dropnan=True)

    return reframed, scaler


def plotting_save_experiment_data(model, history, y_actual, y_predicted):
    # model.save('TheModel_'+ parameters["ID"] +'.h5')
    # del model

    # plot history
    plot_model(model, to_file='experimentOutput\\' + parameters["ID"] + 'model_fig.png', show_shapes=True,
               show_layer_names=True)
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    # print(short_model_summary)

    y_actual = y_actual[:-1]
    y_predicted = y_predicted[1:]

    # save data
    writer = ExcelWriter('experimentOutput\\' + parameters["ID"] + 'results.xlsx')
    df = DataFrame.from_dict(parameters, orient='index')
    df.columns = ['value']
    df.to_excel(writer, 'parameters')
    df = DataFrame(list(zip(y_actual, y_predicted)), columns=['y_actual', 'y_predicted'])
    df.to_excel(writer, 'predicted')
    df = DataFrame(list(zip(history.history['loss'], history.history['val_loss'])), columns=['loss', 'val_loss'])
    df.to_excel(writer, 'loss_history')
    writer.save()
    writer.close()

    # plot history loss
    pyplot.close()
    pyplot.plot(history.history['loss'], label='train_loss')
    pyplot.plot(history.history['val_loss'], label='test_loss')
    # pyplot.plot(history.history['val_mean_absolute_percentage_error'], label='MAPE')
    pyplot.legend()
    pyplot.savefig('experimentOutput\\' + parameters["ID"] + "loss_fig.png")
    pyplot.close()

    # calculate RMSE
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    print('Test RMSE: %.3f' % rmse)
    # calculate MAPE
    MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
    print('Test MAPE: %.3f' % MAPE)
    min_train_loss = min(history.history['loss'])
    min_val_loss = min(history.history['val_loss'])
    print('min train_loss: %.3f' % min_train_loss)
    print('min val_loss: %.3f' % min_val_loss)

    # plot actual vs predicted
    pyplot.plot(y_actual, label='actual')
    pyplot.plot(y_predicted, label='predicted')
    pyplot.legend()
    pyplot.title('RMSE: %.3f' % rmse + " , " + 'MAPE: %.3f' % MAPE)
    pyplot.savefig('experimentOutput\\' + parameters["ID"] + "forcast_fig.png")
    pyplot.close()

    # save to database
    if parameters["save_to_database"]:
        db = mysql.connector.connect(host="localhost", user="root", passwd="mansoura", db="LTLF_data")
        # prepare a cursor object using cursor() method
        cursor = db.cursor()

        sql = """INSERT INTO experiments (experiment_ID, n_days, n_features, n_traindays, n_epochs, n_batch, 
        n_neurons,Dropout, earlystop, RMSE, MAPE, min_train_loss, min_val_loss, Model_summary,comment,optimizer) VALUES ('{}',{},{},{}, {}, {}, 
        {}, {}, {}, {:.4f},{:.4f}, {:.4f}, {:.4f}, '{}', '{}', '{}')""" \
            .format(parameters["ID"], parameters["n_days"], parameters["n_features"], parameters["n_traindays"],
                    parameters["n_epochs"], parameters["n_batch"], parameters["n_neurons"], parameters["Dropout"],
                    parameters["earlystop"], rmse, MAPE, min_train_loss, min_val_loss, short_model_summary,
                    parameters["comment"],parameters["optimizer"])
        try:
            cursor.execute(sql)
            db.commit()
            print("** saved to database")
        except TypeError as e:
            print(e)
            db.rollback()
            print(sql)
            print("** Error saving to database")

        db.close()


def create_fit_model(data, scaler):
    # split into train and test sets
    values = data.values

    train = values[:parameters["n_traindays"], :]
    test = values[parameters["n_traindays"]:, :]
    # split into input and outputs
    n_obs = parameters["n_days"] * parameters["n_features"]
    train_X, train_y = train[:, :n_obs], train[:, -parameters["n_features"]]
    test_X, test_y = test[:, :n_obs], test[:, -parameters["n_features"]]
    # reshape input to be 3D [samples, timesteps, features] for LSTM and CNN
    train_X = train_X.reshape((train_X.shape[0], parameters["n_days"], parameters["n_features"]))
    test_X = test_X.reshape((test_X.shape[0], parameters["n_days"], parameters["n_features"]))
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    # visible1 = Input(shape=(train_X.shape[1], train_X.shape[2]))
    # L1 = LSTM(parameters["n_neurons"])(visible1)
    # L1_1 = Activation('sigmoid')(L1)
    # L2 = Dense(5, activation='relu')(L1_1)
    # output = Dense(1)(L2)
    # model = Model(inputs=visible1, outputs=output)
    # # -------------
    # model = Sequential()
    # model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())
    # model.add(Dense(parameters["n_neurons"], activation='relu'))
    # model.add(Dropout(parameters["Dropout"]))
    # model.add(Dense(1))

    # MaxLoad only
    model = Sequential()
    model.add(LSTM(parameters["n_neurons"], return_sequences=True))
    model.add(Dropout(parameters["Dropout"]))
    model.add(LSTM(parameters["n_neurons"], return_sequences=True))
    model.add(Dropout(parameters["Dropout"]))
    model.add(LSTM(parameters["n_neurons"]))
    model.add(Dropout(parameters["Dropout"]))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(parameters["Dropout"]))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(parameters["Dropout"]))
    model.add(Dense(1))
    model.add(Activation('relu'))

    model.compile(loss='mean_squared_error', optimizer=parameters["optimizer"])

    # fit network
    clbs = None
    if parameters["earlystop"]:
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto')
        clbs = [earlyStopping]

    history = model.fit(train_X, y=train_y, epochs=parameters["n_epochs"], batch_size=parameters["n_batch"],
                        validation_data=(test_X, test_y), verbose=parameters["model_train_verbose"],
                        shuffle=False, callbacks=clbs)

    parameters["n_epochs"] = len(history.history["loss"])

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

    # inv_yhat=1
    # inv_y=1
    plotting_save_experiment_data(model, history, inv_y, inv_yhat)


def run_experiment():
    data, scaler = load_prepare_data()
    create_fit_model(data, scaler)


def main():
    i = 1
    d=1
    op='Adam'
    nn=50
    for bat in [64,256,512]:
        for op in ['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']:
            for drop in [0.2,0.5,0.8]:
                # for nn in [50,100,200]:
                    import datetime
                    now = datetime.datetime.now()
                    parameters["ID"] = now.strftime("%Y%m%d%H%M%S")  # uuid.uuid4().hex
                    parameters["n_days"] = d
                    parameters["n_features"] = 1
                    parameters["n_traindays"] = 365 * 11
                    parameters["n_epochs"] = 1000
                    parameters["Dropout"] = drop
                    parameters["n_batch"] = bat
                    parameters["n_neurons"] = nn
                    parameters["model_train_verbose"] = 2
                    parameters["earlystop"] = True
                    parameters["save_to_database"] = False
                    parameters["optimizer"] = op
                    parameters["comment"] = "Model 6   optimizer=" +  parameters["optimizer"] + " Trail " + str(i)

                    i = i + 1
                    print(parameters["ID"])
                    print(parameters)

                    # https://tensorflow.rstudio.com/blog/time-series-forecasting-with-recurrent-neural-networks.html

                    run_experiment()
                    import gc
                    gc.collect()


main()

# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
