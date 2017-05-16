#!/usr/bin/python
# seqproc_serv.py - hosts a movement detector model over API via flask

# prerequisites: sudo pip install flask-api


# imports
import os
import pickle
import numpy as np
import seqproc_common as sp
from flask import Flask, request, url_for, jsonify
from flask_api import FlaskAPI


# general pathing
rootdir = os.getcwd()
model_file = os.path.join(rootdir,'seqproc', '04_model', 'model.pkl')
model_log = os.path.join(rootdir,'seqproc', '04_model', 'model.log')


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
        if not request.data:
            return jsonify({'movement': False, 'err': 'No JSON data received'}), 400

        # error handling: no resolution tag in JSON
        if not request.data['res']:
            return jsonify({'movement': False, 'err': 'No resolution tag \'res\' in received data'}), 400

        # error handling: no sequence tag in JSON
        if not request.data['seq']:
            return jsonify({'movement': False, 'err': 'No sequence tag \'seq\' in received data'}), 400

        # error handling: sequence length does not match window length
        if(len(request.data['seq']) != windowsize):
            return jsonify({'movement': False, 'err': 'Received sequence length ' + str(len(request.data['seq'])) + ' is different from model window length: ' +  str(windowsize)}), 400

        # read resolution data and point data container to sequence
        json_data = request.data
        res_x = json_data['res']['X']
        res_y = json_data['res']['Y']
        json_data = json_data['seq']

        # build window
        window = sp.window(res_x, res_y, json_data, False)

        # predict and return result
        movement = bool(classifier.predict(window))
        probability = classifier.predict_proba(window)
        return jsonify({'movement': movement, 'probability_movement': probability[0][1], 'probability_no_movement': probability[0][0], 'err': ''}), 200


    # return error when necessary
    except:
        raise


# run server
if __name__ == '__main__':
    app.run(
        debug=True,
        port=5000
    )