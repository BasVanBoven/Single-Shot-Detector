#!/usr/bin/python
# TODO FIXME THIS SCRIPT IS NOT FUNCTIONAL YET FIXME TODO
# seqproc_test.py - sends test requests to the sequence_processor API

# input: first frame of window that is to be tested, without extension


# imports
import os
import requests
import json
import argparse
import numpy as np
import seqproc_common as sp


# handle input arguments
parser = argparse.ArgumentParser(description='Test a Sequence Processor.')
parser.add_argument('video', help='name of video that is to be tested, without extension')
parser.add_argument('-d', '--debug', default=False, action='store_true', help='print debug output')
args = parser.parse_args()


# general pathing
rootdir = os.getcwd()
video = args.video # strip last 5 characters from string
server_url = 'http://127.0.0.1:5000/detect_movement/'
json_folder = os.path.join(rootdir, 'video', 'output', 'json', video)
resolution_csv = os.path.join(rootdir, 'video', 'output', 'resolution', video+'.csv')
model_log = os.path.join(rootdir,'seqproc', '04_model', 'model.log')


# initialize window size, needs to be uneven to make the majority vote function correctly
model_params = np.genfromtxt(model_log, delimiter=',')
windowsize = model_params[1][0]
assert(windowsize % 2 != 0)


# get video resolution
resolution = np.genfromtxt(resolution_csv, delimiter=',', dtype=int)
res_x = resolution[0]
res_y = resolution[1]


# do for each possible non-overlapping window


    # prepare request object
    json_batch =  {'seq': [], 'res': {'X': res_x, 'Y': res_y}}
    for i in range(firstframenumber,firstframenumber+int(windowsize)):
        i_formatted = str(i).zfill(4)
        frame_json_path = os.path.join(json_folder, video+'_'+str(i_formatted)+'.json')
        with open(frame_json_path, 'r') as json_file:
            assert(json_file)
            json_batch['seq'].append(json.load(json_file))


    # send POST request and process result
    headers = {'content-type': 'application/json'}
    if args.debug:
        print json_batch
    r = requests.post(server_url, data=json.dumps(json_batch), headers=headers)
    print(r.text)