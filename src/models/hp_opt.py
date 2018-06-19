from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
import tensorflow as tf
import random as rn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Dense
from keras.layers import BatchNormalization, LSTM
from keras.initializers import Constant, RandomUniform, RandomNormal
from keras.regularizers import l2, l1
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ModelCheckpoint
from keras import backend as K
import os
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import HMP_get_data as hmp
import HAR_get_data as har
from tabulate import tabulate

def fix_seeds():
    """
    Fixes the seeds for reproducibility.
    """
    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

    # fix random seed for reproducibility
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    seed = np.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    return

def get_data():
    mainpath = "/Users/xtian/Documents/Quinn Research Group/accelerometer_research/data/HMP_Dataset"
    dataset="HAR_b"
    if dataset == "HMP":
        activity_labels = hmp.get_labels(mainpath)
        file_dict       = hmp.get_filepaths(mainpath)
        files           = list(file_dict.keys())

        # build training inputs and labels
        # X: each row is a single activity file for one individual's triaxial data
        # y and labels are same shape (label for each timepoint)
        X, y, labels = hmp.build_inputs(
            files,
            activity_labels,
            file_dict)
        # X = pd.read_csv("HMP_X.csv", header=None)
        # y = pd.read_csv("HMP_y.csv", header=None)
        # labels = pd.read_csv("HMP_labels.csv", header=None)


    elif "HAR" in dataset:
        mainpath = "/Users/xtian/Documents/Quinn Research Group/accelerometer_research/data/UCI HAR Dataset"
        data, l, s = har.get_filepaths(mainpath)
        X_t, X_b, y, labels = har.fetch_and_format(data, l , s)
        if dataset == "HAR_t": X = X_t
        elif dataset == "HAR_b": X = X_b

    # shuffle the files so that activities (classes) aren't grouped together
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

    # concatenate each activity file together into one long activity sequence
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

    lstm_one = LSTM(input_shape=(128,3), units={{choice(np.arange(2,512))}},\
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

    lstm_n = LSTM(units={{choice(np.arange(2,512))}},\
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
    # if conditional({{choice([1,3])}}) == 1:
    # model.add(lstm_one)
        # if conditional({{choice([0,1])}}) == 1: model.add(BatchNormalization())
    # elif conditional({{choice([1,2,3])}}) == 2:
    model.add(lstm_input)
    if conditional({{choice([0,1])}}) == 1: model.add(BatchNormalization())
    model.add(lstm_last)
    if conditional({{choice([0,1])}}) == 1: model.add(BatchNormalization())
    # elif conditional({{choice([1,2,3])}}) == 3:
    #     model.add(lstm_input)
    #     if conditional({{choice([0,1])}}) == 1: model.add(BatchNormalization())
    #     model.add(lstm_n)
    #     if conditional({{choice([0,1])}}) == 1: model.add(BatchNormalization())
    #     model.add(lstm_last)
    #     if conditional({{choice([0,1])}}) == 1: model.add(BatchNormalization())
    model.add(Dense(6))
    model.add(Activation('softmax'))

    adam = keras.optimizers.Adam(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, clipnorm=1.)
    rmsprop = keras.optimizers.RMSprop(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, clipnorm=1.)
    sgd = keras.optimizers.SGD(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, clipnorm=1.)

    model.compile(optimizer={{choice(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'nadam', 'adam'])}},
          loss='categorical_crossentropy', metrics=['accuracy'])


    model_saver = ModelCheckpoint('model.{epoch:02d}-{val_loss:.2f}.hdf5')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=0, mode='auto')

    print(trainx_windows.shape)
    results = model.fit(trainx_windows, trainy, epochs=10000, batch_size={{choice(np.arange(32, 450))}}, validation_split=0.2, callbacks=[early_stop, model_saver])
    score, acc = model.evaluate(testx_windows, testy, verbose=1)
    valLoss = result.history['val_mean_absolute_error'][-1]
    parameters = space
    results.append(parameters)

    tab_results = tabulate(results, headers="keys", tablefmt="fancy_grid", floatfmt=".8f")
    weights = model.get_weights()
    print(weights)
    with open('weights.txt', 'a') as weights:
        weights.write("model: {}\n\tlayer: {}\n\tweights: {}\n\tmodel_details: {}".format(model, layer, weights, tab_results))

    print(tab_results)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == "__main__":
    # fix_seeds()

    best_run, best_model = optim.minimize(model=create_model,
        data=get_data,
        algo=tpe.suggest,
        max_evals=3,
        trials=Trials())

    # X_train, X_test, y_train, y_test = get_data()

    # print((X_train.shape))
    # print((X_test.shape))
    # print((y_train.shape))
    # print((y_test.shape))

    # model_dict = create_model(X_train, X_test, y_train, y_test)
    # print(model_dict)
