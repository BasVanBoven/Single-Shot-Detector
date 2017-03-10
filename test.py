#!/usr/bin/python
# train.py - tests a Single Shot Detector


# imports
import os
import sys
import caffe
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google.protobuf import text_format
from caffe.proto import caffe_pb2


# handle input arguments
parser = argparse.ArgumentParser(description='Test a Single Shot Detector.')
parser.add_argument('builddir', help='build (timestamp only) that is to be trained')
args = parser.parse_args()


# global parameters
# current working directory
rootdir = os.getcwd()
# caffe root directory
rootcaffe = '/caffe'
# determines which build to use
builddir = args.builddir
# directory change, meaning all paths after this need to be built from rootdir
os.chdir(rootcaffe)
# Caffe settings
sys.path.insert(0, 'python')
caffe.set_device(0)
caffe.set_mode_gpu()
# MatPlotLib settings
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# open label file
labelmap_file = label_map_file = os.path.join(rootdir, 'builds', builddir, 'includes', 'labelmap.prototxt')
# load detection labels
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)
# check which version of SSD we are running
if os.path.isfile(os.path.join(rootdir, 'builds', builddir, 'ssd300.log')):
    assert(os.path.isfile(os.path.join(rootdir, 'builds', builddir, 'ssd512.log')) == False)
    ssd_version = 300
else:
    assert(os.path.isfile(os.path.join(rootdir, 'builds', builddir, 'ssd512.log')) == True)
    ssd_version = 512


# extracts label names from label file
def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


# finds most recent snapshot
def get_iter_recent():
    max_iter = 0
    for file in os.listdir(os.path.join(rootdir, 'builds', builddir, 'snapshots')):
      if file.endswith(".caffemodel"):
        basename = os.path.splitext(file)[0]
        model_name = "ssd"+str(ssd_version)+"x"+str(ssd_version)
        iter = int(basename.split("{}_iter_".format(model_name))[1])
        if iter > max_iter:
          max_iter = iter
    return max_iter


# processes an image through the SSD network and saves the output to a image file
def process_image(path_input, path_output):
    image = caffe.io.load_image(path_input)
    # forward pass
    detections = net.forward()['detection_out']
    # parse the outputs
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]
    # get detections with confidence higher than a certain threshold
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.3]
    # prepare information for output
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # build output
    currentAxis = plt.gca()
    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f'%(label_name, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    # save output and close all open figures
    plt.savefig(path_output, bbox_inches='tight')
    print ('Processed figure '+path_input)
    plt.close('all')


# paths to model files
iter_recent = get_iter_recent()
model_weights = os.path.join(rootdir, 'builds', builddir, 'snapshots', 'ssd'+str(ssd_version)+'x'+str(ssd_version)+'_iter_'+str(iter_recent)+'.caffemodel')
model_def = os.path.join(rootdir, 'builds', builddir, 'includes', 'ssd'+str(ssd_version), 'deploy.prototxt')
# load Caffe net
net = caffe.Net(model_def, model_weights, caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123]))
# the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_raw_scale('data', 255)
# the reference model has channels in BGR order instead of RGB
transformer.set_channel_swap('data', (2,1,0))
# set net to batch size of 1
image_resize = int(ssd_version)
net.blobs['data'].reshape(1,3,image_resize,image_resize)


# perform all tests in testsets
datacount = 0
for root, dirs, files in os.walk(os.path.join(rootdir, 'testsets')):
    for name in files:
        name, ext = os.path.splitext(name)
        if (ext.lower() == '.jpg'):
            datacount += 1
            output_dirs = root.split("testsets/")[1]
            target_dir = os.path.join(root, '../..', 'builds', builddir, 'testsets_output', output_dirs)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            process_image(os.path.join(root, name+ext), os.path.join(target_dir, name+'.png'))