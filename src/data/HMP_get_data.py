import pandas as pd
import numpy as np
import csv
import os

def get_filepaths(mainfolder):
    """
    Searches a folder for all unique files and compile a dictionary of their paths.

    Parameters
    --------------

    mainfolder: the filepath for the folder containing the data (i.e., filepath to HMP_Dataset)

    Returns
    --------------

    filepaths:
    """
    filepaths = {}

    folders = os.listdir(mainfolder) # data collected into folders for each activity
    for folder in folders:
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "MODEL" not in folder:
            filenames = os.listdir(fpath) # list of files within each activity folder
            for filename in filenames:
                fullpath = fpath + "/" + filename
                filepaths[fullpath] = folder # full filepath = key, activity name = value
    return filepaths

def get_labels(mainfolder):
    """ Creates a dictionary of labels for each unique type of motion """
    labels = {}
    label = 0
    for folder in os.listdir(mainfolder):
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "MODEL" not in folder:
            labels[folder] = label # activity name = keys; numeric label = value
            label += 1
    return labels

def get_data(fp, labels, folders):
    """
    Creates a dataframe for the data in the filepath and creates a one-hot
    encoding of the file's label
    """
    data = pd.read_csv(filepath_or_buffer=fp, sep=' ', names = ["X", "Y", "Z"])
    if data.isnull().values.any(): print("NULL VALUES")
    # arrange entire file into single row
    arr = np.asarray([np.asarray(row) for idx, row in data.iterrows()])
    one_hot = np.zeros(14)
    file_dir = folders[fp]
    label = labels[file_dir]
    one_hot[label] = 1
    return arr, one_hot, label

def build_inputs(files_list, accel_labels, file_label_dict):
    X_seq    = []
    y_seq    = []
    labels = []
    for path in files_list:
        data, target, target_label = get_data(path, accel_labels, file_label_dict)
        X_seq.append(data)
        y_seq.append(list([target for ts in range(len(data))]))
        labels.append(list([target_label for ts in range(len(data))]))
    X_all = np.asarray(X_seq)
    y_all = np.asarray(y_seq)
    labels_all = np.asarray(labels)

    # all_labels = [label for label in all_labels for ts in range(128)]
    return X_all, y_all, labels_all

def write_data(x_data, y_data, labels_data):
    with open('HMP_X.csv', 'w') as HMP_X:
        writer1 = csv.writer(HMP_X)
        for file_data in x_data:
            writer1.writerow(file_data)
    with open('HMP_y.csv', 'w') as HMP_y:
        writer2 = csv.writer(HMP_y)
        for file_onehot in y_data:
            writer2.writerow(file_onehot)
    with open('HMP_labels.csv', 'w') as HMP_labels:
        writer3 = csv.writer(HMP_labels)
        for file_label in labels_data:
            writer3.writerow(file_label)
        # HMP_labels.write("{}".format(labels_data))
    return

if __name__ == '__main__':

    if os.path.isdir("/Users/xtian"):
        mainpath = "/Users/xtian/Documents/Quinn Research Group/accelerometer_research/data/HMP_Dataset"
    else:
        mainpath = "~/Documents"

    activity_labels = get_labels(mainpath)
    file_dict       = get_filepaths(mainpath)
    files           = list(file_dict.keys())

    # build training inputs and labels
    X, y, labels = build_inputs(
        files,
        activity_labels,
        file_dict)
    #
    write_data(X,y,labels)
