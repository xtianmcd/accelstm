from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
# source for the above 4 lines:
# https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
# as stated in post: GPU and tensorflow may both still have backend nuances
# which prevent reproducibility; however, since this code is for hyperparameter
# optimization, should be ok

import sys
import os
# for p in sys.path: print(p)
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
# source: https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
sys.path.append(os.path.dirname(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))) # this is actually what was needed (added accelstm to path)
# sys.path.append("..") # https://stackoverflow.com/questions/30669474/beyond-top-level-package-error-in-relative-import
# for p in sys.path: print(p)

import numpy as np
import tensorflow as tf
import random as rn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.cross_validation import train_test_split
import keras
from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Dense
from keras.layers import BatchNormalization, LSTM
from keras.initializers import Constant, RandomUniform, RandomNormal
from keras.regularizers import l2, l1
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ModelCheckpoint
from keras import backend as K
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from src.data import HMP_get_data
from src.data import HAR_get_data
# from ..data import HAR_get_data as har #https://stackoverflow.com/questions/20075884/python-import-module-from-another-directory-at-the-same-level-in-project-hierar
from tabulate import tabulate


"""
__author__ = "Christian McDaniel"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Christian McDaniel"
__email__ = "clm121@uga.edu"

This file optimizes the hyperparameters of a baseline LSTM model using the TPE
expected improvement algorithm provided by Hyperas from Hyperopt. 

"""

def get_data():
    """
    Calls the necessary functions from the corresponding script in
    accelstm/src/data to read in and reformat the data to be analyzed.

    Returns:
    ----------
    trainx_windows: triaxial windowed training data
    testx_windows:  triaxial windowed testing data
    trainy:         one-hot labels for training the model
    testy:          one-hot labels for evaluating models' predictions


    """
    datapath = "../../data/external/"
    dataset="HAR_t"
    # dataset="HMP"
    # if dataset == "HMP":
    #     activity_labels = hmp.get_labels(mainpath)
    #     file_dict       = hmp.get_filepaths(mainpath)
    #     files           = list(file_dict.keys())
    #
    #     # build training inputs and labels
    #     # X: each row is a single activity file for one individual's triaxial data
    #     # y and labels are same shape (label for each timepoint)
    #     X, y, labels = hmp.build_inputs(
    #         files,
    #         activity_labels,
    #         file_dict)
    #     # X = pd.read_csv("HMP_X.csv", header=None)
    #     # y = pd.read_csv("HMP_y.csv", header=None)
    #     # labels = pd.read_csv("HMP_labels.csv", header=None)


    mainpath = datapath + "HAR"
    data, l, s = HAR_get_data.get_filepaths(mainpath)
    X_t, X_b, X_a, y, labels = HAR_get_data.fetch_and_format(data, l , s)
    if dataset == "HAR_t": X = X_t
    elif dataset == "HAR_b": X = X_b

    # shuffle the participants' data/labels
    shuffle_idx = np.random.permutation(len(X))
    X_shuffle = X[shuffle_idx]
    y_shuffle = y[shuffle_idx]
    labels_shuffle = labels[shuffle_idx]

    # 80/20 split for train and test
    X_train = X_shuffle[:round(0.8*len(X_shuffle))]
    X_test = X_shuffle[round(0.8*len(X_shuffle)):]
    y_train = y_shuffle[:round(0.8*len(y_shuffle))]
    y_test = y_shuffle[round(0.8*len(y_shuffle)):]
    labels_train = labels_shuffle[:round(0.8*len(labels_shuffle))]
    labels_test = labels_shuffle[round(0.8*len(labels_shuffle)):]

    # concatenate each file together into one long activity sequence
    X_traintogether = []
    y_traintogether = []
    X_testtogether = []
    y_testtogether = []
    for datafile in range(len(X_train)):
        X_traintogether.extend(X_train[datafile])
        y_traintogether.extend(y_train[datafile])
    for datafile_test in range(len(X_test)):
        X_testtogether.extend(X_test[datafile_test])
        y_testtogether.extend(y_test[datafile_test])

    X_traintogether = np.asarray(X_traintogether)
    y_traintogether = np.asarray(y_traintogether)
    X_testtogether = np.asarray(X_testtogether)
    y_testtogether = np.asarray(y_testtogether)

    #standardize data: test is fit to training data parameters
    scaler = StandardScaler()
    scaler.fit(X_traintogether)
    X_trainstd = scaler.transform(X_traintogether)
    X_teststd  = scaler.transform(X_testtogether)

    #apply sliding window
    trainx_windows = np.asarray([X_trainstd[i:i+128] for i in range(0,len(X_trainstd)-128,64)])
    trainy_windows = np.asarray([y_traintogether[i:i+128] for i in range(0,len(y_traintogether)-128,64)])
    # window label = label of middle timepoint
    trainy = np.asarray([trainy_windows[window][64] for window in range(len(trainy_windows))])

    testx_windows = np.asarray([X_teststd[i:i+128] for i in range(0,len(X_teststd)-128,64)])
    testy_windows = np.asarray([y_testtogether[i:i+128] for i in range(0,len(y_testtogether)-128,64)])
    testy = np.asarray([testy_windows[window][64] for window in range(len(testy_windows))])

    return trainx_windows, testx_windows, trainy, testy

def create_model(trainx_windows, testx_windows, trainy, testy):
    """
    Used by Hyperas to create and test a unique model by chosing a value at
    each location wwith double-bracketed ranges of values.

    Parameters:
    -------------
    trainx_windows: triaxial windowed training data, returned by get_data()
    testx_windows:  triaxial windowed testing data, returned by get_data()
    trainy:         one-hot labels for training the model, returned by get_data()
    testy:          one-hot labels for evaluating models' predictions, returned
                    by get_data()

    Returns:
    -------------
    loss:   (negative) accuracy of models' predictions
    status: parameter used by Hyperas
    model:  baseline LSTM model with unique hyperparameter selections

    """

    lstm_input = LSTM(input_shape=(128,3), units={{choice(np.arange(2,512))}},\
                    activation={{choice(['softmax', 'tanh', 'sigmoid', 'relu', 'linear'])}},\
                    recurrent_activation={{choice(['softmax', 'tanh', 'sigmoid', 'relu', 'linear'])}},\
                    use_bias={{choice([True, False])}},\
                    kernel_initializer={{choice(['zeros', 'ones', RandomNormal(), RandomUniform(minval=-1, maxval=1, seed=None), Constant(value=0.1), 'orthogonal', 'lecun_normal', 'glorot_uniform'])}},\
                    recurrent_initializer={{choice(['zeros', 'ones', RandomNormal(), RandomUniform(minval=-1, maxval=1, seed=None), Constant(value=0.1), 'orthogonal', 'lecun_normal', 'glorot_uniform'])}},\
                    unit_forget_bias=True,\
                    kernel_regularizer={{choice([None,'l2', 'l1'])}},\
                    recurrent_regularizer={{choice([None,'l2', 'l1'])}},\
                    bias_regularizer={{choice([None,'l2', 'l1'])}},\
                    activity_regularizer={{choice([None,'l2', 'l1'])}},\
                    kernel_constraint=None, recurrent_constraint=None,\
                    bias_constraint=None, dropout={{uniform(0, 1)}},\
                    recurrent_dropout={{uniform(0, 1)}},\
                    return_sequences=True, return_state=False,\
                    go_backwards=False, stateful=False, unroll=False)

    lstm_last = LSTM(units={{choice(np.arange(2,512))}},\
                    activation={{choice(['softmax', 'tanh', 'sigmoid', 'relu', 'linear'])}},\
                    recurrent_activation={{choice(['softmax', 'tanh', 'sigmoid', 'relu', 'linear'])}},\
                    use_bias={{choice([True, False])}},\
                    kernel_initializer={{choice(['zeros', 'ones', RandomNormal(), RandomUniform(minval=-1, maxval=1, seed=None), Constant(value=0.1), 'orthogonal', 'lecun_normal', 'glorot_uniform'])}},\
                    recurrent_initializer={{choice(['zeros', 'ones', RandomNormal(), RandomUniform(minval=-1, maxval=1, seed=None), Constant(value=0.1), 'orthogonal', 'lecun_normal', 'glorot_uniform'])}},\
                    unit_forget_bias=True,\
                    kernel_regularizer={{choice([None,'l2', 'l1'])}},\
                    recurrent_regularizer={{choice([None,'l2', 'l1'])}},\
                    bias_regularizer={{choice([None,'l2', 'l1'])}},\
                    activity_regularizer={{choice([None,'l2', 'l1'])}},\
                    kernel_constraint=None, recurrent_constraint=None,\
                    bias_constraint=None, dropout={{uniform(0, 1)}},\
                    recurrent_dropout={{uniform(0, 1)}},\
                    return_sequences=False, return_state=False,\
                    go_backwards=False, stateful=False, unroll=False)

    model = Sequential()
    model.add(lstm_input)
    if conditional({{choice([0,1])}}) == 1: model.add(BatchNormalization())
    model.add(lstm_last)
    if conditional({{choice([0,1])}}) == 1: model.add(BatchNormalization())
    model.add(Dense(6))
    model.add(Activation('softmax'))

    adam_lr = keras.optimizers.Adam(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, clipnorm=1.)
    rmsprop_lr = keras.optimizers.RMSprop(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, clipnorm=1.)
    sgd_lr = keras.optimizers.SGD(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, clipnorm=1.)

    optims = {{choice(['adam', 'sgd', 'rmsprop', 'adagrad', 'nadam', 'adadelta'])}}
    if optims == 'adam': optim = adam_lr
    elif optims == 'rmsprop': optim = rmsprop_lr
    elif optims == 'sgd': optim = sgd_lr
    else: optim = optims

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optim)

    # model_saver = ModelCheckpoint('model.{epoch:02d}-{val_loss:.2f}.hdf5')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=0, mode='auto')

    # added to collect optimization results
    if 'results' not in globals():
        global results
        results = []

    print(trainx_windows.shape)
    result = model.fit(trainx_windows, trainy, epochs=500, batch_size={{choice(np.arange(32, 450))}}, validation_split=0.2, callbacks=[early_stop]) #, model_saver])
    score, acc = model.evaluate(testx_windows, testy, verbose=1)
    # valLoss = result.history['val_mean_absolute_error'][-1]
    parameters = space
    print(parameters)
    results.append(parameters)

    tab_results = tabulate(results, headers="keys", tablefmt="fancy_grid", floatfmt=".8f")
    weights = model.get_weights()
    # print(weights)
    with open('../../output/hp_opt/weights.txt', 'a+') as model_summ:
        model_summ.write("model: {}\n\tweights:\n{}\n\tmodel_details:\n{}\n\tscore:\t{}".format(model, list(weights), tab_results, acc))

    # print(tab_results)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == "__main__":

    best_run, best_model = optim.minimize(model=create_model,
        data=get_data,
        algo=tpe.suggest,
        max_evals=100000,
        trials=Trials())
