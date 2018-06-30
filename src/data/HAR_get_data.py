import pandas as pd
import numpy as np
import csv
import os

def get_filepaths(mainfolder):
    """
    Retrieves and stores the filepaths containing the accelerometer data, the
    corresponding labels and the corresponding subject number.

    Parameters:
    -------------
    mainfolder: filepath to folder containing all files associated with data

    Returns:
    -------------
    data_filepaths: dict of file paths for each file name
    labels:         labels associated with each pre-windowed data record
    subject:        subjects associated with each pre-windowed data record

    """
    data_filepaths = {}
    labels = []
    subject = []
    files = os.listdir(mainfolder)
    for f in files:
        fpath = mainfolder + "/" + f
        if os.path.isdir(fpath): # data pre-separated into test & train folders
            fpaths = fpath + "/Inertial Signals" # raw signals in this folder
            filenames = os.listdir(fpaths)
            for filename in filenames:
                if "acc" in filename:   # acceleration and gyroscope data
                    fullpath = fpaths + "/" + filename
                    data_filepaths[filename] = fullpath
            if f == "test":
                labels.append(fpath+"/y_test.txt")
                subject.append(fpath+"/subject_test.txt")
            elif f == "train":
                labels.append(fpath+"/y_train.txt")
                subject.append(fpath+"/subject_train.txt")
    return data_filepaths, labels, subject

def fetch_and_format(file_dict, label_fp, subject_fp):
    # initialize data frames and lists
    total_x = pd.DataFrame()
    total_y = pd.DataFrame()
    total_z = pd.DataFrame()
    body_x = pd.DataFrame()
    body_y = pd.DataFrame()
    body_z = pd.DataFrame()
    all_labels = []
    all_subjects = []

    for filename in list(file_dict.keys()):
        data = pd.read_csv(file_dict[filename],
                           delim_whitespace=True,
                           header=None)
        if data.isnull().values.any(): print("NULL VALUES")
        # print(type(data))
        # print(type(filename))
        if "total" in filename:
            if "_x_" in filename:
                total_x = pd.concat([total_x,data], ignore_index=True)
            elif "_y_" in filename:
                total_y = pd.concat([total_y,data], ignore_index=True)
            elif "_z_" in filename:
                total_z = pd.concat([total_z,data], ignore_index=True)
        elif "body" in filename:
            if "_x_" in filename:
                body_x = pd.concat([body_x,data], ignore_index=True)
            elif "_y_" in filename:
                body_y = pd.concat([body_y,data], ignore_index=True)
            elif "_z_" in filename:
                body_z = pd.concat([body_z,data], ignore_index=True)
        # print(total_x.shape)

    print("Initial (windowed axis-wise) data:")
    print(total_x.shape)
    print(total_y.shape)
    print(total_z.shape)
    print(body_x.shape)
    print(body_y.shape)
    print(body_z.shape)

    for fpl in label_fp: # filepath for pre-separated training & testing labels
        labels = open(fpl, 'r')
        label_list = labels.read().splitlines()
        all_labels.extend(label_list)

    for fps in subject_fp: # filepath for pre-separated train & test subjects
        subjects = open(fps, 'r')
        subject_list = subjects.read().splitlines()
        all_subjects.extend(subject_list)

    print(np.asarray(all_subjects).shape)
    print(np.asarray(all_labels).shape)
    print()

    # return pre-windowed single-axis data to continuous time series
    total_x = total_x.iloc[::2].as_matrix().ravel()
    total_y = total_y.iloc[::2].as_matrix().ravel()
    total_z = total_z.iloc[::2].as_matrix().ravel()
    body_x  = body_x.iloc[::2].as_matrix().ravel()
    body_y  = body_y.iloc[::2].as_matrix().ravel()
    body_z  = body_z.iloc[::2].as_matrix().ravel()

    print("halved and concatenated data")
    print(total_x.shape)
    print(total_y.shape)
    print(total_z.shape)
    print(body_x.shape)
    print(body_y.shape)
    print(body_z.shape)
    print()

    # assemble continuous single-axis time series into triaxial time series
    triaxial_t = np.vstack([np.array([total_x[ts], total_y[ts], total_z[ts]]) for ts in range(len(total_x))])
    triaxial_b = np.vstack([np.array([body_x[ts], body_y[ts], body_z[ts]]) for ts in range(len(body_x))])
    sixaxial   = np.vstack([np.array([total_x[ts], total_y[ts], total_z[ts], body_x[ts], body_y[ts], body_z[ts]]) for ts in range(len(triaxial_t))])

    print("Combined axes:")
    print(triaxial_t.shape) # "total" signal (gravity component not removed)
    print(triaxial_b.shape) # "body" signal (gravity component removed)
    print(sixaxial.shape)   # total & body signals
    print()

    all_labels    = all_labels[::2]
    all_subjects  = all_subjects[::2]
    unique_labels = set(all_labels)
    # print(unique_labels)

    # print("labels halved")
    # print(np.asarray(all_labels).shape)
    # print()
    # print(np.asarray(all_subjects).shape)

    # expand the window-wise labels/subject nums to time step-wise granularity
    all_labels   = [label for label in all_labels for ts in range(128)]
    all_subjects = np.array([subject for subject in all_subjects for ts in range(128)])

    # print("expanded")
    # print(np.asarray(all_labels).shape)
    # print()
    # print(np.asarray(all_subjects).shape)

    # Generate one-hot labels
    int_label = 0
    one_hot = np.zeros((len(all_labels),6)) # 6 activity classes
    for u_label in unique_labels:
        for a_label in range(len(all_labels)):
            if u_label == all_labels[a_label]:
                # print(u_label, all_labels[a_label])
                # print(all_labels[a_label])
                # print(one_hot[a_label])
                # print(u_label)
                # print(int(u_label))
                one_hot[a_label][int(u_label)-1] = 1
                # print(one_hot[a_label])
    # print(len(one_hot))
    # indices of data where subject changes (re: train/test split integrity)
    indices = np.where(all_subjects[:-1] != all_subjects[1:])[0]
    # print(indices)
    # print(all_subjects[indices[0]])
    # print(all_subjects[indices[0]+1])
    # print(len(all_labels))

    # subject-wise arrays so train/test sets dont have data from same subject
    start = 0
    triaxial_total = []
    triaxial_body = []
    one_hot_labels = []
    grouped_labels = []
    sixaxial_grouped = []
    # print(type(triaxial_t))
    # print(type(one_hot))
    for index in indices:
        end = index+1
        # print(start)
        # print(end)
        triaxial_total.append(triaxial_t[start:end])
        triaxial_body.append(triaxial_b[start:end])
        sixaxial_grouped.append(sixaxial[start:end])
        one_hot_labels.append(one_hot[start:end])
        grouped_labels.append(all_labels[start:end])
        start = end
    triaxial_total.append(triaxial_t[start:]) # for final interval
    triaxial_body.append(triaxial_b[start:])
    sixaxial_grouped.append(sixaxial[start:])
    one_hot_labels.append(one_hot[start:])
    grouped_labels.append(all_labels[start:])

    triaxial_total = np.asarray(triaxial_total)
    triaxial_body = np.asarray(triaxial_body)
    sixaxial_grouped = np.asarray(sixaxial_grouped)
    one_hot_labels = np.asarray(one_hot_labels)
    grouped_labels = np.asarray(grouped_labels)

    print("Separated by participant:")
    print(triaxial_total.shape)
    print(triaxial_body.shape)
    print(sixaxial_grouped.shape)
    print()

    # sanity check
    t_sum=0
    b_sum=0
    s_sum=0
    o_sum=0
    l_sum=0
    for person in range(30):
        t_sum+=triaxial_total[person].shape[0]
        b_sum+=triaxial_body[person].shape[0]
        s_sum+=sixaxial_grouped[person].shape[0]
        o_sum+=one_hot_labels[person].shape[0]
        grouped_labels[person] = np.asarray(grouped_labels[person])
        l_sum+=grouped_labels[person].shape[0]
    print("... and their summed sizes for comparison:")
    print(t_sum)
    print(b_sum)
    print(s_sum)

    return triaxial_total, triaxial_body, sixaxial_grouped, one_hot_labels, grouped_labels

def write_data(x_total, x_body, y_data, labels_data):
    with open('../../output/test/HAR_X_total.csv', 'w') as HAR_t:
        writer1 = csv.writer(HAR_t)
        for total_data in x_total:
            writer1.writerow(total_data)
    with open('../../output/test/HAR_X_body.csv', 'w') as HAR_b:
        writer2 = csv.writer(HAR_b)
        for body_data in x_body:
            writer2.writerow(body_data)
    with open('../../output/test/HAR_y.csv', 'w') as HAR_y:
        writer3 = csv.writer(HAR_y)
        for subject_onehot in y_data:
            # print(set(subject_onehot))
            writer3.writerow(subject_onehot)
    with open('../../output/test/HAR_labels.csv', 'w') as HAR_labels:
        writer4 = csv.writer(HAR_labels)
        for subject_label in labels_data:
            # print(set(subject_label))
            writer4.writerow(subject_label)
    return

if __name__ == '__main__':

    mainpath = "../../data/external/HAR"

    data, l, s = get_filepaths(mainpath)
    t, b, six, all_1h, all_l = fetch_and_format(data, l , s)

    write_data(t,b,all_1h,all_l)

    print()
    print("|----------------------------------------------|")
    print("| CSV's written to folder accelstm/output/test |")
    print("|----------------------------------------------|")
