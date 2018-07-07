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

import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import BatchNormalization, LSTM, Activation
from keras.initializers import Constant
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler, KFold
import time
import csv
from keras import backend as K
from src.data import HMP_get_data
from src.data import HAR_get_data
from tabulate import tabulate
np.set_printoptions(threshold='nan')

"""
__author__ = "Christian McDaniel"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Christian McDaniel"
__email__ = "clm121@uga.edu"

This file runs 5-fold cross validation on a baseline LSTM model whose
hyperparameters have been optimized by hp_opt.py and more_opt.py.
"""

def get_data():
    """
    Calls the necessary functions from the corresponding script in
    accelstm/src/data to read in and reformat the data to be analyzed.

    Returns:
    ----------
    X_shuffle:      The triaxial accelerometer signals grouped as individual
                    records; shuffled
    y_shuffle:      The one-hot labels corresponding to the data in X_shuffle
    labels_shuffle: The corresponding ground truth labels

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

    return X_shuffle, y_shuffle, labels_shuffle

def multiclass_pred_check(predictions, actual, k, debug=False):
    """
    Calculates the tp, fp, tn and fn statistics from each class. If
    classficiation, takes the index of the maximum value of the one-hot
    prediction as the predicted class. If regression, computes statistics based
    on the provided tolerance value (T). T=0.5 means the nearest integer is the
    predicted class, while T=1.0 means the resulting integer from both the
    floor function or the ceiling function of the prediction are considered
    (more lenient). Stores class-wise statistics as well as sums each together
    for full fold-wise statistics across all classes.
    Parameters:
    -----------------
    predictons: predicted values
    actual:     ground truth values
    k:          num classes
    debug:      Boolean indicating whether or not to print additional
                information; default=False
    Returns:
    -----------------
    tp: number of true  positive predictions
    fp: number of false positive predictions
    fn: number of false negative predictions
    ctp: dictionary with number of true  positive predictions for each class
    cfp: dictionary with number of false positive predictions for each class
    cfn: dictionary with number of false negative predictions for each class
    """
    tp=0
    fp=0
    fn=0
    ctp = {}
    cfp={}
    cfn={}
    classpicks = {}
    for c in range(k):
        tpc=0
        fpc=0
        fnc=0

        tpc = sum([tpc + 1 for i in range(actual.shape[0]) if np.argmax(predictions[i]) == c and np.argmax(actual[i]) == c])
        fpc = sum([fpc + 1 for i in range(actual.shape[0]) if np.argmax(predictions[i]) == c and np.argmax(actual[i]) != c])
        fnc = sum([fnc + 1 for i in range(actual.shape[0]) if np.argmax(predictions[i]) != c and np.argmax(actual[i]) == c])
        if debug: print("class {} tp: {}; fp: {}; fn: {}".format(c, tpc, fpc, fnc))
        tp+=tpc
        fp+=fpc
        fn+=fnc
        ctp[str(c)]=tpc
        cfp[str(c)]=fpc
        cfn[str(c)]=fnc
        classpicks[str(c)]={}
        for cc in range(k):
            csum=0
            classpicks[str(c)][str(cc)]=sum([csum+1 for i in range(actual.shape[0]) if np.argmax(predictions[i])==c])

    if debug: print("fold tp: {}, fp: {}, fn: {}".format(tp, fp, fn))
    return tp, fp, fn, ctp, cfp, cfn

def get_stats(tp, fp, fn, actual):
    """
    Computes performace measures using the correctness measures returned by
    check_preds()
    Parameters:
    -----------------
    tp: number of true  positive predictions returned by check_preds()
    fp: number of false positive predictions returned by check_preds()
    fn: number of false negative predictions returned by check_preds()
    Returns:
    -----------------
    acc:       (tp + tn) / (tp + tn + fp + fn)
    recall:    tp / (tp + fn)
    precision: tp / (tp + fp)
    fscore:    F1 Score, 2 * precision * recall / (precision + recall)
    """
    acc = tp / len(actual)

    recall = 0.0
    precision = 0.0
    fscore = 0.0

    if tp == 0.0:
        if fp == 0.0 and fn == 0.0:
            recall, precision, fscore = 1.0, 1.0, 1.0
        else:
            recall, precision, fscore = 0.0, 0.0, 0.0
    else:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        fscore = 2 * precision * recall / (precision + recall)

    return acc, recall, precision, fscore

def create_model(window_size, n_layers):
    """
    Constructs the model using the optimal parameters selected from hp_opt.py
    and more_opt.py.

    Parameters:
    ------------
    window_size: Number of time steps per window, used by input_shape in the
                 first LSTM layer
    n_layers:    Number of LSTM layers in the network

    Returns:
    ------------
    model: Constructed baseline LSTM model

    """

    # lstm_one = LSTM(input_shape=(window_size,3), units=128,\
    #                 activation='sigmoid',\
    #                 recurrent_activation='tanh',\
    #                 use_bias=True,\
    #                 kernel_initializer='glorot_uniform',\
    #                 unit_forget_bias=True,\
    #                 kernel_regularizer=None,\
    #                 recurrent_regularizer=None,\
    #                 bias_regularizer=None,\
    #                 activity_regularizer=None,\
    #                 kernel_constraint=None, recurrent_constraint=None,\
    #                 bias_constraint=None, dropout=0.5,\
    #                 recurrent_dropout=0.5,\
    #                 return_sequences=False, return_state=False,\
    #                 go_backwards=False, stateful=False, unroll=False)
    lstm_input = LSTM(input_shape=(128,3), units=118,\
                    activation='tanh',\
                    recurrent_activation='tanh',\
                    use_bias=True,\
                    kernel_initializer='orthogonal',\
                    recurrent_initializer='zeros',\
                    unit_forget_bias=True,\
                    kernel_regularizer=None,\
                    recurrent_regularizer=None,\
                    bias_regularizer=None,\
                    activity_regularizer=None,\
                    kernel_constraint=None, recurrent_constraint=None,\
                    bias_constraint=None, dropout=0.303,\
                    recurrent_dropout=0.458,\
                    return_sequences=True, return_state=False,\
                    go_backwards=False, stateful=False, unroll=False)
    # lstm_n = LSTM(units=128,\
    #                 activation='tanh',\
    #                 recurrent_activation='sigmoid',\
    #                 use_bias=True,\
    #                 kernel_initializer='glorot_uniform',\
    #                 unit_forget_bias=True,\
    #                 kernel_regularizer=None,\
    #                 recurrent_regularizer=None,\
    #                 bias_regularizer=None,\
    #                 activity_regularizer=None,\
    #                 kernel_constraint=None, recurrent_constraint=None,\
    #                 bias_constraint=None, dropout=0.0,\
    #                 recurrent_dropout=0.0,\
    #                 return_sequences=True, return_state=False,\
    #                 go_backwards=False, stateful=False, unroll=False)

    lstm_last = LSTM(units=22,\
                    activation='softmax',\
                    recurrent_activation='sigmoid',\
                    use_bias=True,\
                    kernel_initializer='glorot_uniform',\
                    recurrent_initializer=Constant(value=0.1),\
                    unit_forget_bias=True,\
                    kernel_regularizer=None,\
                    recurrent_regularizer='l2',\
                    bias_regularizer=None,\
                    activity_regularizer=None,\
                    kernel_constraint=None, recurrent_constraint=None,\
                    bias_constraint=None, dropout=0.196,\
                    recurrent_dropout=0.073,\
                    return_sequences=False, return_state=False,\
                    go_backwards=False, stateful=False, unroll=False)

    model = Sequential()
    if n_layers == 1: model.add(lstm_one)
    elif n_layers == 2:
        model.add(lstm_input)
        model.add(BatchNormalization())
        model.add(lstm_last)
        model.add(BatchNormalization())
    elif n_layers == 3:
        model.add(lstm_input)
        model.add(lstm_n)
        model.add(lstm_last)
    elif n_layers == 4:
        model.add(lstm_input)
        model.add(lstm_n)
        model.add(lstm_n)
        model.add(lstm_last)

    model.add(Dense(6))
    model.add(Activation('softmax'))
    model.compile(optimizer='nadam',
          loss='categorical_crossentropy', metrics=['accuracy'])


    return model #{'loss': -acc, 'model': model}

def test_model(X, y, params, debug=False):
    """
    Prepares the data and tests the model constructed by create_model().

    Parameters:
    -------------
    X:      data
    y:      target values as one-hot labels
    params: dictionary of parameters for current iteration of grid search
    debug:  prints debuggin statements if True; default=False

    Returns:
    ------------
    Performance statistics from model's predictions on test set; statistics are
    returned by multiclass_pred_check() and get_stats()

    """

    model_acc = []
    model_recall = []
    model_precis = []
    model_fscore = []
    time_elapsed = []
    model_tp = []
    model_fp = []
    model_fn = []
    mc_classwise_acc={}
    mc_classwise_recall={}
    mc_classwise_precis={}
    mc_classwise_fscore={}
    for k in range(6):
            mc_classwise_acc[str(k)]=[]
            mc_classwise_recall[str(k)]=[]
            mc_classwise_precis[str(k)]=[]
            mc_classwise_fscore[str(k)]=[]
    split = 1
    window_size = params['window_size']
    stride = params['stride']
    n_layers = params['n_layers']

    kf = KFold(n_splits=5)
    fold=1
    for train_index, test_index in kf.split(np.zeros(len(X)), np.array([i for i in range(30)])):

        if debug:
            print("training fold {}".format(split))

        # fold-wise train/test split
        print("fold: {}".format(fold))
        X_train=X[train_index]
        X_test=X[test_index]
        y_train=y[train_index]
        y_test=y[test_index]

        #concat together to form one series for train and one for test
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
        print("Train shapes (X, y)")
        print(X_traintogether.shape)
        print(y_traintogether.shape)
        print("Test shapes (X, y)")
        print(X_testtogether.shape)
        print(y_testtogether.shape)
        print()

        # standardize data: test is fit to training data parameters
        scaler = StandardScaler()
        scaler.fit(X_traintogether)
        X_trainstd = scaler.transform(X_traintogether)
        X_teststd  = scaler.transform(X_testtogether)

        #apply sliding window
        trainx_windows = np.asarray([X_trainstd[i:i+window_size] for i in range(0,len(X_trainstd)-window_size,round(stride*window_size))])
        trainy_windows = np.asarray([y_traintogether[i:i+window_size] for i in range(0,len(y_traintogether)-window_size,round(stride*window_size))])
        trainy = np.asarray([trainy_windows[window][round(window_size/2)] for window in range(len(trainy_windows))])

        testx_windows = np.asarray([X_teststd[i:i+window_size] for i in range(0,len(X_teststd)-window_size,round(stride*window_size))])
        testy_windows = np.asarray([y_testtogether[i:i+window_size] for i in range(0,len(y_testtogether)-window_size,round(stride*window_size))])
        testy = np.asarray([testy_windows[window][round(window_size/2)] for window in range(len(testy_windows))])

        print("windowed data (train X, train y, test X, test y)")
        print(trainx_windows.shape)
        print(trainy.shape)
        print(testx_windows.shape)
        print(testy.shape)
        print()

        # Shuffle the windows
        train_shuffle_idx = np.random.permutation(len(trainx_windows))
        X_shuffletrain = trainx_windows[train_shuffle_idx]
        y_shuffletrain = trainy[train_shuffle_idx]

        test_shuffle_idx = np.random.permutation(len(testx_windows))
        X_shuffletest = testx_windows[test_shuffle_idx]
        y_shuffletest = testy[test_shuffle_idx]

        # data_classes={}

        # build and train the model
        model = create_model(window_size, n_layers)

        # model_saver = ModelCheckpoint('model.{epoch:02d}-{val_loss:.2f}.hdf5')
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=0, mode='auto')

        # added to collect optimization results
        if 'results' not in globals():
            global results
            results = []

        result = model.fit(X_shuffletrain, y_shuffletrain, epochs=500, batch_size=140, validation_split=0.2, callbacks=[early_stop]) #, model_saver])

        score, acc = model.evaluate(X_shuffletest, y_shuffletest, verbose=1)
        print("Score: {}; acc: {}".format(score, acc))
        predict = model.predict(testx_windows)

        fold+=1

        tp, fp, fn, ctp, cfp, cfn = multiclass_pred_check(predict, y_shuffletest, 6, debug=True)
        acc, recall, precis, fscore = get_stats(tp, fp, fn, y_shuffletest)

        model_acc.append(acc)
        model_recall.append(recall)
        model_precis.append(precis)
        model_fscore.append(fscore)
        model_tp.append(tp)
        model_fp.append(fp)
        model_fn.append(fn)
        for k in range(6):
            fold_cw_acc, fold_cw_recall, fold_cw_precis, fold_cw_fscore = get_stats(ctp[str(k)], cfp[str(k)], cfn[str(k)], y_shuffletest)
            mc_classwise_acc[str(k)].append(fold_cw_acc)
            mc_classwise_recall[str(k)].append(fold_cw_recall)
            mc_classwise_precis[str(k)].append(fold_cw_precis)
            mc_classwise_fscore[str(k)].append(fold_cw_fscore)

    acc_mean = np.mean(model_acc)
    acc_std = np.std(model_acc)
    fscore_mean = np.mean(model_fscore)
    fscore_std = np.std(model_fscore)
    precis_mean = np.mean(model_precis)
    precis_std = np.std(model_precis)
    recall_mean = np.mean(model_recall)
    recall_std = np.std(model_recall)
    tp_mean = np.mean(model_tp)
    tp_std = np.std(model_tp)
    fp_mean = np.mean(model_fp)
    fp_std = np.std(model_fp)
    fn_mean = np.mean(model_fn)
    fn_std = np.std(model_fn)
    classwise_acc_mean = {}
    classwise_recall_mean = {}
    classwise_precis_mean = {}
    classwise_fscore_mean = {}
    for k in range(6):
        classwise_acc_mean[str(k)] = np.mean(mc_classwise_acc[str(k)])
        classwise_recall_mean[str(k)] = np.mean(mc_classwise_recall[str(k)])
        classwise_precis_mean[str(k)] = np.mean(mc_classwise_precis[str(k)])
        classwise_fscore_mean[str(k)] = np.mean(mc_classwise_fscore[str(k)])

    return acc_mean, acc_std, fscore_mean, fscore_std, tp_mean, fp_mean, fn_mean, tp_std, fp_std, fn_std, recall_mean, recall_std, precis_mean, precis_std, classwise_acc_mean, classwise_precis_mean, classwise_recall_mean, classwise_fscore_mean

if __name__ == "__main__":
    # fix_seeds()
    debug = True
    dataset="HAR_t"
    debug=True
    X, y, l = get_data()
    print(X.shape)
    print(y.shape)


    window_size = 128
    stride      = 0.5
    n_layers    = 2
    params = {}
    params['window_size'] = window_size
    params['stride'] = stride
    params['n_layers'] = n_layers

    print("Testing Model")
    acc_mean, acc_std, fscore_mean, fscore_std, tp_mean, fp_mean, fn_mean, tp_std, fp_std, fn_std, recall_mean, recall_std, precis_mean, precis_std, cw_acc_mean, cw_precis_mean, cw_recall_mean, cw_fscore_mean = test_model(X, y, params)

    if debug: print("writing to file")
    f = open('../../output/final_cv/final_model_results.txt', 'a')
    f.write("Params for model\n")
    f.write("Overall Performance:\n\tAccuracy: {0:.4f}% +/- {0:.4f}\n\tRecall: {0:.4f} +/- {0:.4f}\n\tPrecision: {0:.4f} +/- {0:.4f}\n\tF1Score: {0:.4f} +/- {0:.4f}\n".format(acc_mean*100, acc_std, recall_mean, recall_std, precis_mean, precis_std, fscore_mean, fscore_std))
    f.write("Class-specific Accuracies:\n\t")
    for k in range(6):
        f.write("Class {}: {}%; ".format(k, cw_acc_mean[str(k)]))
    f.write('\n')
    f.write("Class-specific Precisions:\n\t")
    for k in range(6):
        f.write("Class {}: {}%; ".format(k, cw_precis_mean[str(k)]))
    f.write('\n')
    f.write("Class-specific Recall:\n\t")
    for k in range(6):
        f.write("Class {}: {}%; ".format(k, cw_recall_mean[str(k)]))
    f.write('\n')
    f.write("Class-specific FScore:\n\t")
    for k in range(6):
        f.write("Class {}: {}%; ".format(k, cw_fscore_mean[str(k)]))
    f.write('\n')
    f.write("Results: tp:{}, fp:{}, fn:{}\n".format(tp_mean, fp_mean, fn_mean))
    f.write('\n')
    f.write('\n\n')
    f.close()

    print("Params for model")
    print("Overall Performance:\n\tAccuracy: {0:.4f}% +/- {0:.4f}\n\tRecall: {0:.4f} +/- {0:.4f}\n\tPrecision: {0:.4f} +/- {0:.4f}\n\tF1Score: {0:.4f} +/- {0:.4f}\n".format(acc_mean*100, acc_std, recall_mean, recall_std, precis_mean, precis_std, fscore_mean, fscore_std))
    print("Class-specific Accuracies:\n\t")
    for k in range(6):
        print("Class {}: {}%; ".format(k, cw_acc_mean[str(k)]))
    print()
    print("Class-specific Precisions:\n\t")
    for k in range(6):
        print("Class {}: {}%; ".format(k, cw_precis_mean[str(k)]))
    print()
    print("Class-specific Recall:\n\t")
    for k in range(6):
        print("Class {}: {}%; ".format(k, cw_recall_mean[str(k)]))
    print()
    print("Class-specific FScore:\n\t")
    for k in range(6):
        print("Class {}: {}%; ".format(k, cw_fscore_mean[str(k)]))
    print()
    print("Results: tp:{}, fp:{}, fn:{}\n".format(tp_mean, fp_mean, fn_mean))
    print()
    print()
