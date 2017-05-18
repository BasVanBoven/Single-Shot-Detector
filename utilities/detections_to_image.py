#!/usr/bin/python
# detections_to_image.py - embed bounding boxes to an image based on input image and detection json


# imports
import os
import sys
import json
import argparse
import urllib2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from scipy.misc import imread


# handle input arguments
parser = argparse.ArgumentParser(description='Embed bounding boxes to an image based on input image and detection JSON.')
parser.add_argument('image', help='path to input image')
parser.add_argument('json', help='path to json')
parser.add_argument('output', help='path to output image')
parser.add_argument('-v', '--verbose', default=False, action='store_true', help='print all detections, not only top ones')
args = parser.parse_args()


# processes an image through the SSD network and saves the output to a image file
def process_image(path_image, path_json, path_output):

    # labelcolors
    labelcolors = {'cabin': 0, 'forearm': 1, 'upperarm': 2, 'wheelbase': 3, 'attachment-bucket': 4, 'attachment-breaker': 5, 'manual-crouching': 6, 'manual-earthrod': 7}

    # pyplot settings
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # load input
    if path_image.startswith('http'):
        image = io.imread(path_image)
    else:
        image = io.imread(os.path.join(os.getcwd(), path_image))
    if path_json.startswith('http'):
        response = urllib2.urlopen(path_json)
        detections = json.load(response)
    else:
        with open(os.path.join(os.getcwd(), path_json), 'rb') as json_data:
            detections = json.load(json_data)

    # image setup: detection colors, axes and background
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    currentAxis = plt.gca()
    plt.imshow(image, alpha=1)

    # translate json output to nice array format
    num_detections = len(detections['body']['predictions'][0]['classes'])
    det = []
    for i in range(num_detections):
        current = detections['body']['predictions'][0]['classes'][i]
        det.append([current['cat'], current['prob'], current['bbox']['xmin'], current['bbox']['ymin'], current['bbox']['xmax'], current['bbox']['ymax']])

    # calculate maximum confidences
    max_conf = {'cabin': 0, 'forearm': 0, 'upperarm': 0, 'wheelbase': 0, 'attachment-bucket': 0, 'attachment-breaker': 0, 'manual-crouching': 0, 'manual-earthrod': 0}
    for i in range(num_detections):
        max_conf[det[i][0]] = max(max_conf[det[i][0]], det[i][1])

    # strip non-maximum detections
    det_stripped = []
    for i in range(num_detections):
        if ((det[i][1] >= max_conf[det[i][0]]) or args.verbose or det[i][0] in ['manual-crouching' , 'manual-earthrod']):
            det_stripped.append(det[i])

    # build output
    for i in range(len(det_stripped)):
        score = det_stripped[i][1]
        label_name = det_stripped[i][0]
        xmin = int(round(det_stripped[i][2]))
        ymin = int(round(det_stripped[i][3]))
        xmax = int(round(det_stripped[i][4]))
        ymax = int(round(det_stripped[i][5]))
        display_txt = '%s: %.2f'%(label_name, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[labelcolors[det_stripped[i][0]]]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

    # save output and close all open figures
    plt.savefig(os.path.join(os.getcwd(), path_output), bbox_inches='tight')
    print ('Processed figure '+path_image)
    plt.close('all')


# perform all tests in testsets
process_image(args.image, args.json, args.output)