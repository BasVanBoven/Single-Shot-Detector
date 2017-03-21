#!/usr/bin/python
# benchmark.py - benchmarks a Single Shot Detector


# imports
import os
import sys
import caffe
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google.protobuf import text_format
from caffe.proto import caffe_pb2


# handle input arguments
parser = argparse.ArgumentParser(description='Benchmark a Single Shot Detector.')
parser.add_argument('builddir', help='build (timestamp only) that is to be benchmarked')
parser.add_argument('-c', '--conf', type=float, default=0.4, help='confidence a detection must have to count')
parser.add_argument('-o', '--overlap', type=float, default=0.5, help='overlap a detection must have with the ground truth to count')
args = parser.parse_args()


# global parameters
# current working directory
rootdir = os.getcwd()
# caffe root directory
rootcaffe = '/caffe'
# test directory
testdir = 'testsets_new'
testpath = os.path.join(rootdir, testdir)
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
          # ugly hack: hardcode the model iteration to use
          #max_iter = 50000
    return max_iter


# processes an image through the SSD network and compares the output
def process_image(path_input, total_detections, successful_detections):
    image = caffe.io.load_image(path_input)
    plt.imshow(image, alpha=0)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
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
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= args.conf]
    # prepare prediction results
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    # prepare ground truth tree
    name, ext = os.path.splitext(path_input)
    assert os.path.exists(os.path.join(testpath, name + '.xml'))
    tree = ET.parse(os.path.join(testpath, name + '.xml'))
    tree_root = tree.getroot()
    # for each ground truth bounding box
    for object in tree_root.findall('object'):
        total_detections = total_detections + 1
        found = False
        gt_xmin = int(object.find('bndbox').find('xmin').text)
        gt_ymin = int(object.find('bndbox').find('ymin').text)
        gt_xmax = int(object.find('bndbox').find('xmax').text)
        gt_ymax = int(object.find('bndbox').find('ymax').text)
        # iterate over all predictions
        for i in xrange(top_conf.shape[0]):
            if (object.find('name').text == top_labels[i]):
                # extract prediction results
                pred_xmin = int(round(top_xmin[i] * image.shape[1]))
                pred_ymin = int(round(top_ymin[i] * image.shape[0]))
                pred_xmax = int(round(top_xmax[i] * image.shape[1]))
                pred_ymax = int(round(top_ymax[i] * image.shape[0]))
                # determine edges of overlap
                left = max(min(gt_xmin,gt_xmax),min(pred_xmin,pred_xmax))
                right = min(max(gt_xmin,gt_xmax),max(pred_xmin,pred_xmax))
                top = max(min(gt_ymin,gt_ymax),min(pred_ymin,pred_ymax))
                bottom = min(max(gt_ymin,gt_ymax),max(pred_ymin,pred_ymax))
                # calculate overlap percentage
                surface_overlap = (right-left)*(bottom-top)
                surface_f2 = abs(gt_xmax-gt_xmin)*abs(gt_ymax-gt_ymin)
                overlap_percentage = float(surface_overlap) / float(surface_f2)
                print overlap_percentage
                if overlap_percentage > args.overlap:
                    found = True
        if (found == True):
            successful_detections = successful_detections + 1
    print ('Processed figure '+path_input)
    return(total_detections, successful_detections)
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
for root, dirs, files in os.walk(testpath):
    for directory in dirs:
        total_detections = 0
        successful_detections = 0
        print ('Processing testset '+directory+' ...')
        for subroot, subdirs, subfiles in os.walk(os.path.join(testpath, directory)):
            for name in subfiles:
                name, ext = os.path.splitext(name)
                if (ext.lower() == '.jpg'):
                    returntuple = process_image(os.path.join(subroot, name+ext), total_detections, successful_detections)
                    total_detections = returntuple[0]
                    successful_detections = returntuple[1]
        accuracy = float(successful_detections) / float(total_detections)
        print ('Accuracy for '+directory+': '+ str(accuracy * 100))