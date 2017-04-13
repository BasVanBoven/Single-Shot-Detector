#!/usr/bin/python
# seqproc_serv.py - hosts a movement detector model over API via flask.

# input: a video folder containing json files: bounding boxes, tags and video resolution
# prerequisites: sudo pip install flask-api


# imports
import os
import sys
import json
import random
import pickle
import argparse
import numpy as np
from flask_api import FlaskAPI, status, exceptions
from flask import request, url_for, jsonify


# handle input arguments
parser = argparse.ArgumentParser(description='Process input data for training a Sequence Processor.')
parser.add_argument('-d', '--debug', default=False, action='store_true', help='print debug output')
args = parser.parse_args()


# general pathing
rootdir = os.getcwd()
model_file = os.path.join(rootdir,'seqproc', '05_model', 'model.pkl')
model_log = os.path.join(rootdir,'seqproc', '05_model', 'model.log')


# initialize window size, needs to be uneven to make the majority vote function correctly
model_params = np.genfromtxt(model_log, delimiter=',')
windowsize = model_params[1][0]
assert(windowsize % 2 != 0)


# initialize model server
app = FlaskAPI(__name__)
classifier = pickle.load(open(model_file, 'rb'))
@app.route('/detect_movement/', methods=['POST'])
def detect_movement():


    # on API call
    try:

        # error handling: no JSON data received
        if not json_data:
            return jsonify({'movement': False, 'err': 'No JSON data received'}), 400

        # error handling: no resolution tag in JSON
        if not json_data['res']:
            return jsonify({'movement': False, 'err': 'No resolution tag \'res\' in received data'}), 400

        # error handling: no sequence tag in JSON
        if not json_data['seq']:
            return jsonify({'movement': False, 'err': 'No sequence tag \'seq\' in received data'}), 400

        # error handling: sequence length does not match window length
        if(len(json_data['seq']) != windowsize):
            return jsonify({'movement': False, 'err': 'Received sequence length ' + str(len(json_data)) + ' is different from model window length: ' +  str(windowsize)}), 400

        # read resolution data and point data container to sequence
        json_data = request.data
        xres = json_data['res']['X']
        yres = json_data['res']['Y']
        json_data = json_data['seq']

        # convert json frame sequence to window
        window_undifferenced = []
        # do for each frame
        for frame in json_data:
            # get the strongest detection for each category
            frame_data = json.load(open(os.path.join(boxes_folder, frame), 'r'))
            object_dict = {}
            # do for each object
            for detected_object in frame['body']['predictions'][0]['classes']:
                category = detected_object['cat']
                if category in object_dict:
                    if object_dict[category]['prob'] < detected_object['prob']:
                        object_dict[category] = detected_object
                else:
                    object_dict[category] = detected_object
            # take only excavator parts to the sequence processor
            ordering = ['cabin', 'forearm', 'upperarm', 'wheelbase', 'attachment-bucket', 'attachment-breaker']
            # write highest detections to array
            for item in ordering:
                if item in object_dict:
                    # translate to new format
                    obj = object_dict[item]
                    C_X = ((obj['bbox']['xmax'] - obj['bbox']['xmin']) / 2 + obj['bbox']['xmin']) / res_x
                    C_Y = ((obj['bbox']['ymin'] - obj['bbox']['ymax']) / 2 + obj['bbox']['ymax']) / res_y
                    W = (obj['bbox']['xmax'] - obj['bbox']['xmin']) / res_x
                    H = (obj['bbox']['ymin'] - obj['bbox']['ymax']) / res_y
                    Conf = obj['prob']
                    window_undifferenced.extend([C_X, C_Y, W, H, Conf])
                else:
                    # when an excavator part is not detected
                    window_undifferenced.extend([0,0,0,0,0])
        # make sure the list length is correct
        assert(len(window_undifferenced) == windowsize * 30)
        # difference each list item
        window_differenced = []
        for i in range(0, len(window)):
            # difference when not the first frame, otherwise, fill zeroes
            if i < 30:
                window_differenced.extend([window_undifferenced[i], 0])
            else:
                window_differenced.extend([window_undifferenced[i], window_undifferenced[i] - window_undifferenced[i-30]])
        # make sure the list length is still correct
        assert(len(window_differenced) == windowsize * 60)

        # predict and return result
        movement = bool(classifier.predict(window_differenced))
        return jsonify({'movement': movement, 'err': ''}), 200


    # return error when necessary
    except:
        raise


# run server
if __name__ == '__main__':
    app.run(debug=True)