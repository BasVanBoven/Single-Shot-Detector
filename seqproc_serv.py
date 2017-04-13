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
#parser.add_argument('-r', '--round', type=int, default=5, help='number of significant digits in return')
args = parser.parse_args()


# general pathing
rootdir = os.getcwd()
model_file = os.path.join(rootdir,'seqproc', '05_model', 'model.pkl')
model_log = os.path.join(rootdir,'seqproc', '05_model', 'model.log')


# initialize window size, needs to be uneven to make the majority vote function correctly
model_params = np.genfromtxt(model_log, delimiter=',')
windowsize = model_params[1][0]
print windowsize
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
        window = []
        # do for each frame
        for frame in json_data:
            # get the strongest detection for each category
            frame_data = json.load(open(os.path.join(boxes_folder, frame), 'r'))
            object_dict = {}
            for detected_object in frame['body']['predictions'][0]['classes']:
                category = detected_object['cat']
                if category in object_dict:
                    if object_dict[category]['prob'] < detected_object['prob']:
                        object_dict[category] = detected_object
                else:
                    object_dict[category] = detected_object
            # take only excavator parts to the sequence processor
            ordering = ['cabin', 'forearm', 'upperarm', 'wheelbase', 'attachment-bucket', 'attachment-breaker']
            for item in ordering:
                if item in object_dict:
                    # translate to new format
                    obj = object_dict[item]
                    C_X = ((obj['bbox']['xmax'] - obj['bbox']['xmin']) / 2 + obj['bbox']['xmin']) / res_x
                    C_Y = ((obj['bbox']['ymin'] - obj['bbox']['ymax']) / 2 + obj['bbox']['ymax']) / res_y
                    W = (obj['bbox']['xmax'] - obj['bbox']['xmin']) / res_x
                    H = (obj['bbox']['ymin'] - obj['bbox']['ymax']) / res_y
                    Conf = obj['prob']
                    detections.extend([C_X, C_Y, W, H, Conf])
                else:
                    # when an excavator part is not detected
                    detections.extend([0,0,0,0,0])
            # push detections to window
            #TODO
            with open(os.path.join(output_frames, video+'.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(detections)

        # frame csvs -> window csvs
        for filename in sorted(os.listdir(output_frames)):
            # make sure the window contains more than one frame
            assert(window.size > 31)
            # open output file, i.e., the window csv
            with open(os.path.join(output_windows, filename), 'wb') as f:
                writer = csv.writer(f)
                # initialize window buffer
                windowbuffer = []
                classificationbuffer = []
                bufferlength = 0
                # do for each frame, i.e., each line
                for i in range(window.shape[0]):
                    # otherwise, add the frame to the window
                    bufferlength = bufferlength + 1
                    classificationbuffer.extend([window[i][0]])
                    # do for each frame value
                    for j in range(0, window[i][:].shape[0]):
                        # difference when not the first frame, otherwise, fill zeroes
                        if bufferlength == 1:
                            windowbuffer.extend(np.append([window[i][j]], [0]))
                        else:
                            windowbuffer.extend(np.append([window[i][j]], [window[i][j] - window[i-1][j]]))
                    # make sure the window size
                    assert(bufferlength == windowsize)

        '''# Transform N x 60 matrix intro 1 X (N*60) row vector for inference
        X = X.stack().to_frame().T

        # Reset row index and column names
        X = X.reset_index(drop=True)
        X.columns = range(X.shape[1])'''

        # predict and return result
        movement = bool(classifier.predict(X)[0])
        return jsonify({'movement': movement, 'err': ''}), 200


    # return error when necessary
    except:
        raise


# run server
if __name__ == '__main__':
    app.run(debug=True)