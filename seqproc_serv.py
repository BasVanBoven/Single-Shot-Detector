'''
Module: sequence_processor.py
Version: 0.2
Python Version: 2.7.13
Authors: Hakim Khalafi, <>
Description:

    This module hosts a movement detector model over API via flask.
    Prerequisites: sudo pip install flask-api
'''

## Imports
import os
import sys
import json
import random
import numpy as np
import pandas as pd
import pickle

from flask_api import FlaskAPI, status, exceptions
from flask import request, url_for, jsonify

app = FlaskAPI(__name__)

## Configurations
N = 5+1 # The addition is for the base frame, as we are differencing and will end up with one less in the end
rounding = 5
script_folder = os.path.realpath('.')
model_folder = '/models/'
model_file = 'AdaBoostClassifier_N5_t20170407-135017_F2_984.pkl'
total_path = script_folder + model_folder + model_file
clf = pickle.load(open(total_path, 'rb'))

@app.route('/detect_movement/', methods=['POST'])
def detect_movement():
    '''API call to detect movement in sequence of object detection JSONs'''
    try:
        ## Read
        json_data = request.data

        resolution = json_data['res']
        xres = resolution['X']
        yres = resolution['Y']

        json_data = json_data['seq']

        #print(json_data)

        ## Faulty inputs
        # No seq tag in JSON
        if not json_data:
            return jsonify({'movement': False, 'err': 'No sequence tag \'seq\' in received data'}), 400

        # Seq len differs from N
        seq_len = len(json_data)
        if(seq_len != N):
            return jsonify({'movement': False, 'err': 'Received sequence length ' + str(seq_len) + ' is different from required N: ' +  str(N)}), 400

        ## Main processing
        # Random result placeholder
        result = []
        for frame in json_data:
            object_dict = {}
            for detected_object in frame["body"]["predictions"][0]["classes"]:
                # print(detected_object['cat'] + '\t' + str(detected_object['prob']) + '\t' + str(detected_object['bbox']))

                category = detected_object['cat']

                # Save best achieved confidence score in dictionary
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
                    #Reversed here cause erroneously reversed in input data..
                    C_Y = ((obj['bbox']['ymin'] - obj['bbox']['ymax']) / 2 + obj['bbox']['ymax']) / yres
                    W = (obj['bbox']['xmax'] - obj['bbox']['xmin']) / xres
                    H = (obj['bbox']['ymin'] - obj['bbox']['ymax']) / yres
                    Conf = obj['prob']

                    data_array.extend([C_X,C_Y,W,H,Conf])
                else:
                    data_array.extend([0,0,0,0,0]) #If object not present, its elems are empty

            #print(data_array)

            rounded = [np.round(float(i), rounding) for i in data_array]

            result.append(rounded)
            object_dict.clear()

        df = pd.DataFrame(result)
        X = df
        X = pd.concat([X.diff()[1:],X],axis=1)[1:] # Get differenced, augment with absolute, and drop first row of NAs

        # Transform N x 60 matrix intro 1 X (N*60) row vector for inference
        X = X.stack().to_frame().T

        # Reset row index and column names
        X = X.reset_index(drop=True)
        X.columns = range(X.shape[1])

        movement = bool(clf.predict(X)[0])

        return jsonify({'movement': movement, 'err': ''}), 200

    except:
        raise

if __name__ == '__main__':
    app.run(debug=True)