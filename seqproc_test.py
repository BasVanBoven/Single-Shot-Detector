'''
Module: 04_sequence_tester.ipynb
Version: 0.2
Python Version: 2.7.13
Authors: Hakim Khalafi, <>
Description:

    This module sends test requests to sequence_processor API
'''

## Imports

import os
import os.path
from pprint import pprint
import csv
import numpy as np
import pandas as pd
import json
import random
import requests
import json

## Configurations

N = 5+1
sequence_processor_url = 'http://127.0.0.1:5000/detect_movement/'

script_folder = os.path.realpath('.')
json_folder = '/json/'
resolution_folder = '/resolution/'
video_name = 'record_20170307_batch2_000_2'
test_folder = '/' + video_name + '/'

resolution_csv = video_name + '.csv'

res_df = pd.read_csv(script_folder + resolution_folder + resolution_csv, header=None)
xres = int(res_df[0][0])
yres = int(res_df[1][0])

total_path = script_folder + json_folder + test_folder


## Test random sequence of N jsons, picked from test_folder

files = [name for name in os.listdir(total_path) if os.path.isfile(total_path + name)]

batches = int(len(files) / N) # Truncate last samples as they wont be a full batch

chosen_batch = random.randint(0, batches)

json_batch =  {'seq': [], 'res': {'X': xres, 'Y': yres}}

for j in range(0,N):
    file = files[chosen_batch*N + j]
    print(file)

    with open(total_path + file, 'r') as json_in:
        json_batch['seq'].append(json.load(json_in))

#with open('result.json', 'w') as fp:
    #json.dump(json_batch, fp)


headers = {'content-type': 'application/json'}

r = requests.post(sequence_processor_url, data=json.dumps(json_batch), headers=headers)
#dictionary = json.loads(r.text)
#df = pd.DataFrame(dictionary['err'])
print(r.text)