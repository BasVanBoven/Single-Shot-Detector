'''
Module: seqproc_train.py
Authors: Hakim Khalafi, Bas van Boven
Description:

    This module takes as input directories of .json files.
    Each JSON describes bounding box contents of 1 image frame from a Single Shot Detector hosted over Deep Detect API
    The output is a master data .CSV file which contains summarized info of each JSON as a row in the CSV.
    Each object has designated columns in the CSV, where the highest confidence object is used in case of several detections.

    This module takes a master data csv file as produced by 01_convert_json_to_csv_master.ipynb
    Each entry of this CSV file is then batched together into N points, and turned into a row in a reshaped .csv
    This "Windowing" allows for classification of sequences of data which allows for motion detection.
    Each previous datapoints classification gets to vote on which category the windowed batch belongs to.
    The majority (mode) vote wins.

    This module takes the windowed master data file created by 02_split.ipynb and performs classification on it.
    Some information is printed such as confusion matrices, F2 scores, cross validations, data size.
    Finally the model is saved for re-use by sequence processor API.
'''


## Imports
import json
import os
import pickle
import time
import itertools
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.stats import mode
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,fbeta_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier


## Methods
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


## Configurations
N = 9
script_folder = os.path.realpath('.')
json_folder = '/video/output/json/'
video_names  = get_immediate_subdirectories(script_folder + json_folder)
csv_folder = '/video/output/tags/'
resolution_folder = '/video/output/resolution/'
master_data = '/seqproc/master_data/'
out_folder = '/seqproc/N_master_windowed/'
total_path = script_folder + master_data


## Augmenting
augmenting = True
if augmenting:
    augment_filename = '_mirroring'
else:
    augment_filename = ''


## Convert
for video_name in video_names:

    sequence_csv = video_name + '.csv'
    resolution_csv = video_name + '.csv'

    total_path = script_folder + json_folder + '/' + video_name + '/'

    # Resolution dataframe
    res_df = pd.read_csv(script_folder + resolution_folder + resolution_csv, header=None)
    xres = int(res_df[0][0])
    yres = int(res_df[1][0])

    # Tags dataframe, containing SeqProc tags for each video (0,1,2 for nodig, dig and unusable)
    seq_df = pd.read_csv(script_folder + csv_folder + sequence_csv, header=None)
    seq_df = pd.concat([seq_df[0] - 1,seq_df[1]], axis=1) #Remove 1 from first column, for code count (0,1... vs 1,2... )
    counter = 0
    nshape = seq_df.shape[0]
    print('Processing: ' + video_name)

    # Use tags to determine how to label each json
    current_tag = seq_df.loc[0][1]
    change_index = float('Inf')
    if(counter+1 < nshape):
        change_index = seq_df.loc[1][0]

    #print("Current tag is " + str(current_tag) + " changes at index: " + str(change_index))

    # For each json file in video directory
    for idx,filename in enumerate(os.listdir(total_path)):
            with open(total_path + filename, 'r') as json_in:
                # Upon reaching a json where tagging changes
                if(idx == change_index):
                    counter = counter+1
                    current_tag = seq_df.loc[counter][1]
                    if(counter+1 < nshape):
                        change_index = seq_df.loc[counter+1][0]

                #print(idx)
                data = json.load(json_in)
                object_dict = {}
                for detected_object in data["body"]["predictions"][0]["classes"]:
                    #print(detected_object['cat'] + '\t' + str(detected_object['prob']) + '\t' + str(detected_object['bbox']))

                    category = detected_object['cat']

                    if category in object_dict:
                        if(object_dict[category]['prob'] < detected_object['prob']):
                            object_dict[category] = detected_object
                    else:
                        object_dict[category] = detected_object
                #pprint(object_dict)
                data_array = []
                ordering = ["cabin", "forearm", "upperarm","wheelbase","attachment-bucket","attachment-breaker"]
                for idx,item in enumerate(ordering):
                    if item in object_dict:
                        obj = object_dict[item]
                        C_X = ((obj['bbox']['xmax'] - obj['bbox']['xmin']) / 2 + obj['bbox']['xmin']) / xres
                         # ymin and ymax Reversed below cause erroneously reversed in input data..
                        C_Y = ((obj['bbox']['ymin'] - obj['bbox']['ymax']) / 2 + obj['bbox']['ymax']) / yres
                        W = (obj['bbox']['xmax'] - obj['bbox']['xmin']) / xres
                        H = (obj['bbox']['ymin'] - obj['bbox']['ymax']) / yres
                        Conf = obj['prob']

                        # Place future data augmentation logic here. As example see X mirroring:

                        # Mirroring data, comment out if not using this for mirroring!
                        if(augmenting):
                            C_X = 1 - C_X

                        data_array.extend([C_X,C_Y,W,H,Conf])
                    else:
                        data_array.extend([0,0,0,0,0])

                data_array.extend([current_tag])
                #pprint(data_array)

                with open(script_folder + '/seqproc/master_data/' + video_name  + '_master_data' + augment_filename + '.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
                    rounded = [float(i.round(decimals=5)) for i in np.array(data_array)]

                    # If unusable tag, do not write to csv. Ideally this logic would be placed earlier before calculations..
                    if(current_tag != 2): writer.writerow(rounded)
                object_dict.clear()
print("Done")


## Produce differenced data (Y2-Y1) and augment with current Y2 so we get [Y2-Y1, Y2] for each row (windowed)
## We do this to retain information about relative movement from previous frame

for filename in os.listdir(total_path):
    path = os.path.join(total_path, filename)
    if os.path.isdir(path):
        # skip directories
        continue

    df = pd.read_csv(total_path + filename, header=None)
    X = df
    X = X.drop(30, axis = 1)
    X = pd.concat([X.diff()[1:],X],axis=1)[1:] # Get differenced data, augment with absolute data, and drop first row of NAs
    Y = df[30][1:] #Drop first row of NAs due to differencing

    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)

    # Amount of lines in new csv
    # Note we lose the last few datapoints that dont make it into a whole batch..
    lines = int(X.shape[0] / N)

    # For each batch
    for i in range(lines):
        votes = []
        subline = pd.DataFrame()

        # Create windowed batch of N datapoints, which will become one row in new reshaped csv.
        for j in range(N):
            line_nr = i*N + j
            votes.append(Y[line_nr])
            subline = pd.concat([subline,X.iloc[line_nr]])

        # Label voting
        seq_label = int(mode(votes)[0][0])

        subline = subline.reset_index(drop=True)

        subline.loc[len(subline)] = seq_label
        with open(total_path + out_folder + 'md_reshaped_n' + str(N) +  '.csv' , 'a') as f:
            subline.transpose().to_csv(f, header=False, index=False, float_format='%.5f')