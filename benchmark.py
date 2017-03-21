#!/usr/bin/python
# benchmark.py - benchmarks a Single Shot Detector


# imports
import os
os.environ['GLOG_minloglevel'] = '2'
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
parser.add_argument('-i', '--iter', type=int, default=0, help='use a specific model iteration')
parser.add_argument('-c', '--conf', type=float, default=0.4, help='confidence a detection must have to count')
parser.add_argument('-o', '--overlap', type=float, default=0.5, help='overlap a detection must have with the ground truth to count')
args = parser.parse_args()


# global parameters
# current working directory
rootdir = os.getcwd()
# caffe root directory
rootcaffe = '/caffe'
# test directory
testdir = 'testsets_benchmark'
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
        # if an iteration is specified manually, use that
        if args.iter != 0:
            max_iter = args.iter
    return max_iter


# processes an image through the SSD network and compares the output
def process_image(path_input, gt_total, gt_ok):
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
        if gt_total.has_key(object.find('name').text) == False:
            gt_total[object.find('name').text] = 0
            gt_ok[object.find('name').text] = 0
        gt_total[object.find('name').text] = gt_total[object.find('name').text] + 1
        gt_total['total'] = gt_total['total'] + 1
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
                if overlap_percentage > args.overlap:
                    found = True
        if (found == True):
            gt_ok[object.find('name').text] = gt_ok[object.find('name').text] + 1
            gt_ok['total'] = gt_ok['total'] + 1
    return(gt_total, gt_ok)
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
        gt_total = {'total': 0}
        gt_ok = {'total': 0}
        #print ('Processing testset '+directory+' ...')
        for subroot, subdirs, subfiles in os.walk(os.path.join(testpath, directory)):
            for name in subfiles:
                name, ext = os.path.splitext(name)
                if (ext.lower() == '.jpg'):
                    gt_total, gt_ok = process_image(os.path.join(subroot, name+ext), gt_total, gt_ok)
        accuracy = float(gt_ok['total']) / float(gt_total['total'])
        print ('Accuracy for '+directory+': '+ str(accuracy * 100))
        for i in gt_total:
            if i != 'total':
                print i, gt_total[i], gt_ok[i], float(gt_ok[i]) / float(gt_total[i])